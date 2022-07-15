# 2020-05-12 Bokyong Seo<gkbaturu@gmail.com>
import sys
import signal
import click
import sentry_sdk
import settings
import messages
from model.irt import controller


@click.command()
@click.option('-m', '--months', type=click.IntRange(-48, -1), default=-24, help='모델 트레이닝 할 데이터 기간. 단위 월, -48 ~ -1 범위. 배치 실행일로부터 계산.')
@click.option('-g', '--gpu', type=str, default="-1", help='사용할 GPU ID. CUDA_VISIBLE_DEVICES 환경변수 참고. CPU를 사용할 경우 -1.')
def start_batch(months, gpu):
    sentry_sdk.init(dsn=settings.SENTRY_DSN)

    def shutdown_handler(signum, frame):
        sentry_sdk.capture_exception(RuntimeError(
            messages.MSG_SHUTDOWN.format(signal.Signals(signum).name)))
        sys.exit(1)

    _set_signal_handler(shutdown_handler)

    sentry_sdk.add_breadcrumb(category='IRT Batch')
    sentry_sdk.capture_message(messages.MSG_START.format('IRT batch'))

    try:
        controller.batch(months, gpu)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        sys.exit(1)

    sentry_sdk.capture_message(messages.MSG_END.format('IRT batch'))

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
