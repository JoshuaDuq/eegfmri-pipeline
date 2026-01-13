import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    if hasattr(sys.stdout, 'buffer') and (not hasattr(sys.stdout, 'encoding') or sys.stdout.encoding != 'utf-8'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer') and (not hasattr(sys.stderr, 'encoding') or sys.stderr.encoding != 'utf-8'):
        import io
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import subprocess
import mne
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from pipeline.config import update_config, get_config_keyval, get_specific_config
from pipeline.preprocess import run_bads_detection, synchronize_bad_channels_across_runs
from pipeline.ica import run_ica_label
from pipeline.stats import collect_preprocessing_stats
from pipeline.tfr import custom_tfr
from mne_bids_pipeline._logging import logger


def run_pipeline_task(task, config_file):
    update_config(config_file, {"task": task})
    
    logger.info(msg=f"👍 Running preprocessing pipeline for task: {task} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if get_config_keyval(config_file, "use_pyprep"):
        run_bads_detection(**get_specific_config(config_file, "pyprep"))
        synchronize_bad_channels_across_runs(
            bids_path=get_config_keyval(config_file, "bids_root"),
            task=task,
            subjects=get_config_keyval(config_file, "subjects"),
        )
    
    if get_config_keyval(config_file, "use_icalabel"):
        cmd = [
            sys.executable,
            "-c",
            f"from mne_bids_pipeline._main import main; import sys; sys.argv = ['mne_bids_pipeline', '--config={config_file}', '--steps=init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter,preprocessing/_05_regress_artifact,preprocessing/_06a1_fit_ica']; main()",
        ]
        logger.info(f"Running MNE-BIDS pipeline command: {' '.join(cmd)}")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        
        if result.stdout:
            logger.info(f"MNE-BIDS pipeline stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"MNE-BIDS pipeline stderr: {result.stderr}")
            
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout if result.stdout else "No error output captured"
            logger.error(f"MNE-BIDS pipeline failed with return code {result.returncode}. Error output: {error_msg}")
            raise RuntimeError(f"MNE-BIDS pipeline step failed with return code {result.returncode}: {error_msg}")
    else:
        cmd = [
            sys.executable,
            "-c",
            f"from mne_bids_pipeline._main import main; import sys; sys.argv = ['mne_bids_pipeline', '--config={config_file}', '--steps=init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter,preprocessing/_05_regress_artifact,preprocessing/_06a1_fit_ica,preprocessing/_06a2_find_ica_artifacts.py']; main()",
        ]
        logger.info(f"Running MNE-BIDS pipeline command: {' '.join(cmd)}")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        
        if result.stdout:
            logger.info(f"MNE-BIDS pipeline stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"MNE-BIDS pipeline stderr: {result.stderr}")
            
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout if result.stdout else "No error output captured"
            logger.error(f"MNE-BIDS pipeline failed with return code {result.returncode}. Error output: {error_msg}")
            raise RuntimeError(f"MNE-BIDS pipeline step failed with return code {result.returncode}: {error_msg}")
    
    if get_config_keyval(config_file, "use_icalabel"):
        run_ica_label(**get_specific_config(config_file, "icalabel"))
    
    cmd = [
        sys.executable,
        "-c",
        f"from mne_bids_pipeline._main import main; import sys; sys.argv = ['mne_bids_pipeline', '--config={config_file}', '--steps=preprocessing/_07_make_epochs,preprocessing/_08a_apply_ica,preprocessing/_09_ptp_reject']; main()",
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    if result.returncode != 0:
        logger.error(f"MNE-BIDS pipeline failed with error: {result.stderr}")
        raise RuntimeError(f"MNE-BIDS pipeline step failed: {result.stderr}")
    
    collect_preprocessing_stats(
        bids_path=get_config_keyval(config_file, "bids_root"),
        pipeline_path=get_config_keyval(config_file, "deriv_root"),
        task=task,
    )
    
    if get_config_keyval(config_file, "custom_tfr"):
        custom_tfr(**get_specific_config(config_file, "custom_tfr"))
    
    logger.info(f"✅ Preprocessing completed for task: {task}. Run 'eeg-pipeline features' for feature extraction.")


def run_all_tasks(config_file):
    tasks = get_config_keyval(config_file, "tasks_to_process")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    msg = f"Welcome! 👋 The pipeline will be run sequentially for the following tasks: {', '.join(tasks)} using the data from the following BIDS folder:{get_config_keyval(config_file, 'bids_root')}."
    logger.info(msg)
    
    for idx, task in enumerate(tasks):
        update_config(config_file, {"task": task})
        deriv_root = get_config_keyval(config_file, "deriv_root")
        if not os.path.exists(deriv_root):
            os.makedirs(deriv_root)
        
        log_path = os.path.join(deriv_root, task + "_pipeline.log")
        
        py_command = (
            f"import sys; "
            f"sys.path.insert(0, r'{project_root}'); "
            f"from scripts.run_pipeline import run_pipeline_task; "
            f"run_pipeline_task(r'{task}', r'{config_file}')"
        )
        
        if get_config_keyval(config_file, "log_type") == "file":
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            with open(log_path, 'w', encoding='utf-8', errors='replace') as log_file:
                process = subprocess.Popen(
                    [sys.executable, "-c", py_command],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    encoding='utf-8',
                    errors='replace'
                )
                exit_code = process.wait()
            msg = f"👍 Running pipeline for task {idx+1} out of {len(tasks)}: {task} and logging to {log_path}"
            logger.info(msg)
        else:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            process = subprocess.Popen(
                [sys.executable, "-c", py_command],
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            exit_code = process.wait()
            print(f"👍 Running pipeline for task {idx+1} out of {len(tasks)}: {task} and logging to console")
        if exit_code != 0:
            logger.error(f"❌ Pipeline failed for task {task}. Check the log file for details.")
            sys.exit(1)
        else:
            logger.info(f"✅ Pipeline completed successfully for task {task}.")


def main():
    if len(sys.argv) < 2:
        print("ERROR! Usage: python run_pipeline.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    mne.set_log_level("ERROR")
    matplotlib.pyplot.set_loglevel("ERROR")
    
    run_all_tasks(config_file)


if __name__ == "__main__":
    main()

