# 2020-12-01 Bokyong Seo<kay.seo@storypot.io>
import sys
import signal
import click
import sentry_sdk
from datetime import datetime
from db.connector import DBConnector
from model.fm import controller
import settings
import rules
import messages


@click.command()
@click.option('-s', '--subject', type=click.Choice(['K', 'E', 'M', 'S', 'N'], case_sensitive=False), required=True,
              help='트레이닝 할 과목. 국어-K / 영어-E / 수학-M / 사회-S / 과학-N')
def start_batch(subject):
    """
    FM 배치 시작점

    Parameters
    ----------
    subject: str
        트레이닝할 과목(커멘드라인 옵션)

    Returns
    -------
    None
    """
    sentry_sdk.init(dsn=settings.SENTRY_DSN)

    def shutdown_handler(signum, frame):
        sentry_sdk.capture_exception(RuntimeError(
            messages.MSG_SHUTDOWN_SIGNAL.format(signal.Signals(signum).name)))
        sys.exit(1)

    _set_signal_handler(shutdown_handler)

    db_connector = DBConnector(**settings.SETTING_DB_SETTINGS)
    ai_settings = settings.load(subject, 'B', db_connector)
    ai_rule = rules.load(subject, 'B', db_connector)

    sentry_sdk.add_breadcrumb(category='FM Batch')
    start_datetime = datetime.now()
    sentry_sdk.capture_message(messages.MSG_START.format(subject, start_datetime))

    try:
        controller.batch(ai_settings, ai_rule, DBConnector(**settings.STATISTICS_DB_SETTINGS))
    except Exception as e:
        sentry_sdk.capture_exception(e)
        sys.exit(1)

    end_datetime = datetime.now()
    running_time = end_datetime - start_datetime

    sentry_sdk.capture_message(messages.MSG_END.format(subject, end_datetime, running_time))
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
