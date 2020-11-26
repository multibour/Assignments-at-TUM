import pytest
import sys
import os


@pytest.yield_fixture(scope="session")
def silence_printing(pytestconfig):
    # Disable capturing so printing can be disabled
    # See https://github.com/pytest-dev/pytest/issues/1599
    # Unfortunately that does not seem to work yet (promising fix in the dev-code) 
    # -> the -s option is really our only choice here
    # The way to really reduce the output would be to use --tb=line
    # Additionally, we should use pytest-timeout and set a timeout globally
    # Hence we should run:
    # pytest -q -s --disable-warning --show-capture=no --tb=line --timeout 60 --junitxml=<path> tests/
    capmanager = pytestconfig.pluginmanager.getplugin('capturemanager')
    context = capmanager.suspend_global_capture()

    # Redirect stdout to /dev/null
    old_out = sys.stdout
    old_err = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    yield silence_printing

    # Restore standard functionality
    sys.stdout = old_out
    sys.stderr = old_err
    capmanager.resume_global_capture()

@pytest.fixture()
def basic_security( silence_printing):
    return

