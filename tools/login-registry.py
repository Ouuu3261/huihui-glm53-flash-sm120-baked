import os
import subprocess

result = subprocess.run(['crane', 'auth', 'login', 'ghcr.io', '--username', os.environ['GITHUB_ACTOR'], '--password-stdin'],
                        input=os.environ['GITHUB_TOKEN'].encode(), capture_output=True)
if result.returncode:
    raise SystemExit('GHCR login failed')
print('GHCR workflow authentication ready')
