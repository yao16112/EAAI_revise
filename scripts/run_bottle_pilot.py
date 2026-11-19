"""One authorized 60-epoch bottle pilot: pause after epoch 1, resume in a fresh process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tool.experiment import digest_file,write_json
from scripts.recompute_predictions import recompute


def run(output, protocol_path):
    output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(protocol_path.read_text())
    for name,expected in protocol['code_sha256'].items():
        if digest_file(ROOT/name)!=expected:
            raise ValueError('Pilot code differs from frozen draft: '+name)
    configuration={**protocol['training'],'dataset':'mvtec','category':['bottle'],'seed':111}
    write_json(output/'pilot_configuration.json',{'configuration':configuration,'evidence':'PILOT_NOT_FORMAL',
        'protocol':str(protocol_path.resolve()),'protocol_sha256':digest_file(protocol_path),
        'pid':os.getpid(),'python':sys.executable,'cwd':str(ROOT),'start_unix':time.time()})
    started=time.time()
    try:
        def stage(name,extra):
            config={**configuration,'save_path':str(output/name),**extra}
            command=[sys.executable,'-u',str(ROOT/'main.py')]
            for key,value in config.items():
                command.append('--'+key)
                if isinstance(value,list):command.extend(map(str,value))
                else:command.append(str(value).lower() if isinstance(value,bool) else str(value))
            write_json(output/(name+'_command.json'),{'argv':command,'cwd':str(ROOT)})
            with (output/(name+'.log')).open('x') as stream:
                process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                write_json(output/(name+'_process.json'),{'pid':process.pid,'log':str(output/(name+'.log'))})
                code=process.wait()
            write_json(output/(name+'_exit.json'),{'exit_code':code})
            if code:raise RuntimeError(f'{name} exited {code}')
        stage('epoch1',{'stop_after_epoch':1})
        pause=json.loads((output/'epoch1/PAUSED.json').read_text())
        assert pause['completed_epochs']==1 and pause['global_step']==105
        stage('resumed',{'resume':pause['checkpoint']})
        result=recompute(output/'resumed/predictions/bottle')
        assert result['count']==83
        result.update(evidence='PILOT_60_EPOCHS_NOT_FORMAL',total_epochs=60,seed=111,
                      resume_from=pause,elapsed_seconds=time.time()-started,
                      paper_protocol='PAPER REPRODUCTION UNVERIFIED')
        write_json(output/'PILOT_COMPLETE.json',result)
        text=f'''# Pilot Results

**Pilot: COMPLETE** — MVTec AD bottle, seed111, fixed final epoch60.

This is an engineering pilot, excluded from the official five-seed statistics. PAPER REPRODUCTION UNVERIFIED.

- Training: 209 normal images; batch2; input1022; LR5e-4; 60 total epochs.
- Paused after epoch1 / step105, resumed in a new process and output directory; completed epochs2–60.
- Full test: 20 normal + 63 abnormal images; 83 lossless exports.
- Recomputed all evaluator metrics from exports: maximum difference {result['maximum_metric_difference']}.
- Elapsed: {result['elapsed_seconds']:.1f} seconds. Both training stage exit codes: 0.
- Metrics (%): `{json.dumps(result['metrics']['bottle'],ensure_ascii=False)}`.
- Exact commands, PIDs, checkpoints and logs: `{output}`.
- Completed evidence: `{output/'PILOT_COMPLETE.json'}`.

Resume caveat: separate L1 tests restored all RNG/counters/optimizer state, but CUDA backward produced small numerical differences; stepwise bitwise equality is not promised. See IMPLEMENTATION_REPORT.md.

ENGINEERING READY: epoch recovery and export tested. PAPER PROTOCOL VERIFIED: NO. MULTI-SEED READY: formal launch remains disabled pending protocol decisions.
'''
        (ROOT/'experiments/statistical_reliability/PILOT_RESULTS.md').write_text(text)
    except Exception:
        write_json(output/'PILOT_FAILED.json',{'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started})
        with (ROOT/'experiments/statistical_reliability/PILOT_RESULTS.md').open('a') as stream:
            stream.write(f'\nPilot failed. Inspect `{output}/PILOT_FAILED.json` and stage logs.\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args()
    run(args.output,args.protocol)
