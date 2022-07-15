# 2020-08-06 Keonhoon Lee<khlee@dbdiscover.com>
import sys
import signal
import click
import sentry_sdk
import settings
import messages
from model.dkt import controller


@click.command()
@click.option('-c', '--course', type=click.Choice(['thd', 'ath'], case_sensitive=False), required=True, help='트레이닝 할 코스. 매일학교공부-thd / 성취도평가-ath')
@click.option('-sk', '--skill', type=click.Choice(['achieve', 'lesson', 'lecture', 'cognitive', 'content', 'difficulty', 'ptype'], case_sensitive=False), required=True, help='트레이닝 할 스킬')
@click.option('-m', '--months', type=click.IntRange(-48, -1), default=-18, help='모델 트레이닝 할 데이터 기간. 단위 월, -48 ~ -1 범위. 배치가 실행되는 주 시작일(일요일)로부터 계산.')
@click.option('-g', '--gpu', type=str, default="-1", help='사용할 GPU ID. CUDA_VISIBLE_DEVICES 환경변수 참고. CPU를 사용할 경우 -1.')
def start_batch(course, skill, months, gpu):
    sentry_sdk.init(dsn=settings.SENTRY_DSN)

    def shutdown_handler(signum, frame):
        sentry_sdk.capture_exception(RuntimeError(
            messages.MSG_SHUTDOWN_SIGNAL.format(signal.Signals(signum).name)))
        sys.exit(1)

    _set_signal_handler(shutdown_handler)

    if skill not in settings.BATCH_OPTIONS[course]:
        sentry_sdk.capture_exception(RuntimeError(messages.MSG_INVAILD_SKILL.format(skill, course)))
        sys.exit(1)

    sentry_sdk.add_breadcrumb(category='DKT Batch')
    sentry_sdk.capture_message(messages.MSG_START.format(
        'DKT batch for {} / {}'.format(course, skill)))

    try:
        controller.batch(course, skill, months, gpu)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        sys.exit(1)

    sentry_sdk.capture_message(messages.MSG_END.format(
        'DKT batch for {} / {}'.format(course, skill)))

    sys.exit(0)


def _set_signal_handler(shutdown_handler):
    """
    OS 시그널 핸들러 설정

    Parameters
    ----------
    shutdown_handler: function
        셧다운 헨들러

    Returns
    -------
    None
    """
    signal.signal(signal.SIGABRT, shutdown_handler)
    signal.signal(signal.SIGFPE, shutdown_handler)
    signal.signal(signal.SIGILL, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGSEGV, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGQUIT, shutdown_handler)


if __name__ == '__main__':
    start_batch()
