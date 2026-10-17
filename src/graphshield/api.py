"""Local-only FastAPI interface. No label endpoint and no automatic model fitting."""
from contextlib import asynccontextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Literal

from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .paths import ROOT
from .service_store import Conflict, Missing, Store
from .serving import Bundle, InvestigationService
from .monitoring import RequestMetrics, workflow_health

Identifier = Annotated[str,Field(min_length=1,max_length=128,pattern=r'^[A-Za-z0-9_.:-]+$')]
Step = Annotated[StrictInt,Field(ge=1,le=1000000)]
ModelName = Literal['activity_lgb','graph_lgb','node2vec_lgb','mlp','graphsage','graph_only','graph_device']
Policy = Literal['tuned_threshold','budget_only']
Status = Literal['new','in_review','escalated','closed']


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Transaction(Contract):
    transaction_id: Identifier
    sender_id: Identifier
    receiver_id: Identifier
    amount: str
    step: Step
    available_step: Step
    type: Literal['TRANSFER'] = 'TRANSFER'

    @field_validator('amount')
    @classmethod
    def exact_money(cls,value):
        if len(value)>40:
            raise ValueError('Amount is too long')
        try:
            amount=Decimal(value)
            if not amount.is_finite() or amount<0 or amount>=Decimal('1e22') or amount!=amount.quantize(Decimal('.01')):
                raise ValueError('Amount must be finite, nonnegative and exact to two decimals')
        except InvalidOperation as exc:
            raise ValueError('Invalid decimal amount') from exc
        return format(amount,'.2f')

    @model_validator(mode='after')
    def arrival(self):
        if self.available_step<self.step:
            raise ValueError('Availability cannot precede the event step')
        return self


class Observation(Contract):
    observation_id: Identifier
    account_id: Identifier
    device_id: Identifier
    step: Step
    available_step: Step

    @model_validator(mode='after')
    def arrival(self):
        if self.available_step<self.step:
            raise ValueError('Availability cannot precede the event step')
        return self


class IngestRequest(Contract):
    world_id: Identifier
    idempotency_key: Identifier
    transactions: list[Transaction] = Field(default_factory=list,max_length=5000)
    device_observations: list[Observation] = Field(default_factory=list,max_length=5000)


class WorldRequest(Contract):
    world_id: Annotated[str,Field(pattern=r'^demo_[A-Za-z0-9_-]{1,60}$')]
    label: Annotated[str,Field(min_length=1,max_length=80)]


class ScoreRequest(Contract):
    world_id: Identifier
    cutoff_step: Step
    model: ModelName | None = None
    policy: Policy = 'tuned_threshold'


class ReplayRequest(Contract):
    world_id: Identifier
    model: ModelName | None = None
    policy: Policy = 'tuned_threshold'


class ReviewRequest(Contract):
    request_id: Identifier
    expected_version: Annotated[StrictInt,Field(ge=0)]
    status: Status
    note: Annotated[str,Field(min_length=1,max_length=2000)]

    @field_validator('note')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():
            raise ValueError('A review note is required')
        return value.strip()


def create_app(bundle_name='serving-v1',database='artifacts/phase6/demo.sqlite',root=ROOT,service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service or InvestigationService(Bundle(bundle_name,root),Store(database,root))
        yield

    app = FastAPI(title='GraphShield AI local investigation API',version='0.7.0',lifespan=lifespan,
                  docs_url=None,redoc_url=None,openapi_url='/api/v1/openapi.json')
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','testserver'])
    app.state.metrics = RequestMetrics()

    async def request_metrics(request,call_next):
        import time
        started=time.perf_counter()
        status=500
        try:
            response=await call_next(request)
            status=response.status_code
            return response
        finally:
            route=request.scope.get('route')
            template=getattr(route,'path','unmatched')
            method=request.method if request.method in ('GET','POST','HEAD','OPTIONS','PUT','PATCH','DELETE') else 'OTHER'
            app.state.metrics.record(method,template,status,(time.perf_counter()-started)*1000)

    @app.middleware('http')
    async def local_boundary(request,call_next):
        if request.method not in ('GET','HEAD','OPTIONS'):
            origin=request.headers.get('origin')
            if origin and origin!=str(request.base_url).rstrip('/'):
                return JSONResponse({'detail':'Cross-origin writes are disabled'},status_code=403)
            try:
                if int(request.headers.get('content-length','0'))>4_000_000:
                    return JSONResponse({'detail':'Request body exceeds 4 MB'},status_code=413)
            except ValueError:
                return JSONResponse({'detail':'Invalid content length'},status_code=400)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        response.headers['Cache-Control']='no-store'
        return response

    # Register last so rejected browser origins and oversized requests are counted too.
    app.middleware('http')(request_metrics)

    @app.exception_handler(Conflict)
    async def conflict_handler(request,exc):
        return JSONResponse({'detail':str(exc)},status_code=409)

    @app.exception_handler(Missing)
    async def missing_handler(request,exc):
        return JSONResponse({'detail':str(exc.args[0])},status_code=404)

    @app.exception_handler(ValueError)
    async def value_handler(request,exc):
        return JSONResponse({'detail':str(exc)},status_code=400)

    def svc(request):
        return request.app.state.service

    @app.get('/api/v1/health')
    def health(request:Request):
        return {'status':'ready','bundle_version':svc(request).bundle.version,'experimental':True}

    @app.get('/api/v1/catalog')
    def catalog(request:Request):
        return dict(svc(request).bundle.catalog(),stored_worlds=svc(request).store.worlds())

    @app.get('/api/v1/monitoring')
    def monitoring(request:Request):
        return {'requests':app.state.metrics.snapshot(),'workflow':workflow_health(svc(request)),
                'bundle_version':svc(request).bundle.version}

    @app.post('/api/v1/worlds',status_code=201)
    def create_world(body:WorldRequest,request:Request):
        store=svc(request).store
        try:
            existing=store.world(body.world_id)
            if existing['label']!=body.label:
                raise Conflict('Network ID already has a different label')
        except Missing:
            store.register_world(body.world_id,body.label,'User-supplied synthetic demo events')
        return store.world(body.world_id)

    @app.post('/api/v1/ingest')
    def ingest(body:IngestRequest,request:Request):
        if body.world_id in {w['world_id'] for w in svc(request).bundle.meta['worlds']}:
            raise Conflict('Bundled development networks are immutable; create a demo network for supplied events')
        return svc(request).store.ingest(body.world_id,body.idempotency_key,
                                        [r.model_dump() for r in body.transactions],
                                        [r.model_dump() for r in body.device_observations])

    @app.post('/api/v1/score')
    def score(body:ScoreRequest,request:Request):
        return svc(request).score(body.world_id,body.cutoff_step,body.model,body.policy)

    @app.post('/api/v1/replays',status_code=202)
    def start_replay(body:ReplayRequest,request:Request,tasks:BackgroundTasks):
        service=svc(request)
        job=service.request_replay(body.world_id,body.model,body.policy)
        if service.claim(job['replay_id']):
            def background():
                # Error state is persisted by the runner and visible in the dashboard.
                try:
                    service.run_replay(job['replay_id'],already_claimed=True)
                except Exception:
                    import logging
                    logging.getLogger('graphshield').exception('Replay failed')
            tasks.add_task(background)
        return job

    @app.get('/api/v1/replays')
    def replay_list(request:Request):
        return {'items':svc(request).store.replays()}

    @app.get('/api/v1/replays/{replay_id}')
    def replay_detail(replay_id:str,request:Request):
        return svc(request).store.replay(replay_id)

    @app.get('/api/v1/cases')
    def cases(request:Request,world_id:str|None=None,model:ModelName|None=None,status:Status|None=None,
              replay_id:str|None=None,policy:Policy|None=None,limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
        return svc(request).store.list_cases(world_id,model,status,replay_id,limit,offset,policy,svc(request).bundle.version)

    @app.get('/api/v1/cases/{case_id}')
    def case(case_id:str,request:Request):
        return svc(request).store.case(case_id)

    @app.post('/api/v1/cases/{case_id}/reviews')
    def review(case_id:str,body:ReviewRequest,request:Request):
        return svc(request).store.review(case_id,body.request_id,body.expected_version,body.status,body.note)

    static=Path(__file__).parent/'web'
    app.mount('/assets',StaticFiles(directory=static),name='assets')

    @app.get('/',include_in_schema=False)
    def dashboard():
        return FileResponse(static/'index.html')

    return app
