"""Resume this OLMo2 run once its upstream activation upload is complete."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

from affect_geometry.persistence.exp3_hub import hub_api

PROJECT = Path('/workspace/affect-geometry')
LOGS = PROJECT / 'logs/olmo2-exp3-analysis-full-v1-analysis'
AUDIT = PROJECT / 'outputs/exp3/olmo2/recovery/source_audit.json'
PREFIX = 'activations/rq3-activations-v1'
REPO = 'kokolamba/rq3-olmo2'
audit = json.loads(AUDIT.read_text())
bundle = next((PROJECT / 'outputs/exp3/olmo2/downloads').glob('*/activations/rq3-activations-v1'))
checkpoint = json.loads((bundle / 'checkpoints/run.json').read_text())
required = set(checkpoint['artifacts']) | {'checkpoints/run.json'}
api = hub_api()
print('OLMo2 paused: upstream activation bundle is incomplete.', flush=True)
print('Checking every 60 seconds; analysis will restart automatically after all required files are available.', flush=True)
while True:
    try:
        revision = api.repo_info(REPO, repo_type='dataset', revision='main').sha
        files = set(api.list_repo_files(REPO, repo_type='dataset', revision=revision))
        missing = sorted(name for name in required if f'{PREFIX}/{name}' not in files)
        checked = datetime.now(timezone.utc).isoformat()
        audit.update(revision=revision, checked_at=checked,
                     status='waiting_for_complete_remote_bundle' if missing else 'remote_inventory_complete',
                     missing_artifacts=missing)
        temporary = AUDIT.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(audit, indent=2) + '\n')
        temporary.replace(AUDIT)
        print(f'{checked}: {len(missing)} required files missing from {REPO}@{revision[:12]}', flush=True)
        if not missing:
            print('Remote inventory complete. Restarting analysis with normal checksum and provenance validation.', flush=True)
            os.environ['HF_HOME'] = str(PROJECT / 'outputs/exp3/hf_cache')
            result = subprocess.run(['bash', str(LOGS / 'launch.sh')], cwd=PROJECT)
            raise SystemExit(result.returncode)
    except Exception as error:
        print(f'Upstream check failed ({type(error).__name__}); retrying in 60 seconds.', flush=True)
    time.sleep(60)
