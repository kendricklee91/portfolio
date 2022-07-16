# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from celery import Celery

# Celery Worker 시작점(Project file)
# Celery Worker는 이 파일과 /model/fm 디렉토리 아래의 파일만 참조하도록 되어 있음.

sentry_sdk.init(
    dsn="http://ec72d7c66e224c7bbeba12023c039d86@183.110.210.106:9000/2",
    integrations=[CeleryIntegration()],
    environment="CELERY"
)  # Sentry 초기화

celery_app = Celery(
    'tasks',
    broker='amqp://guest:guest@rabbitmq:5672//',
    include=['model.fm.tasks']
)  # Celery 초기화
