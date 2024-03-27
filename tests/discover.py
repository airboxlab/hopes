import os
import sys
import webbrowser
from unittest import TextTestRunner, defaultTestLoader


def run(*, exit=True):
    test_suite = defaultTestLoader.discover("tests")
    result = TextTestRunner(verbosity=2).run(test_suite)

    if os.getenv("TEST_SKIP_FATAL", "false") == "true":
        if result.skipped:
            sys.exit("FAIL: tests were skipped and TEST_SKIP_FATAL is set")

    success = result.wasSuccessful()

    if exit:
        sys.exit(not success)

    return success


def coverage():
    from coverage import Coverage

    cov = Coverage()
    cov.start()

    cwd = os.getcwd()
    success = run(exit=False)
    os.chdir(cwd)

    cov.stop()
    cov.html_report()
    cov.xml_report()
    cov.report()
    webbrowser.open("htmlcov/index.html")

    sys.exit(not success)
