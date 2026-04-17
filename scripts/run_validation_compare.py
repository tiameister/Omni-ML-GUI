"""Run configurable CV protocol comparisons via main.py.

This script sets environment variables to override output root and CV config,
then invokes main.py for each selected protocol under validation_compare/.
"""
import argparse
import os
import subprocess
import sys

from utils.logger import configure_logging, get_logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable or 'python'
LOGGER = get_logger(__name__)


def _parse_protocols(raw: str) -> list[str]:
    allowed = {'kfold', 'repeated', 'nested'}
    protocols = [p.strip().lower() for p in raw.split(',') if p.strip()]
    invalid = [p for p in protocols if p not in allowed]
    if invalid:
        raise ValueError(f"Unsupported protocol(s): {', '.join(invalid)}. Allowed: kfold,repeated,nested")
    if not protocols:
        raise ValueError('At least one protocol must be specified.')
    return protocols


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run CV protocol comparison with main.py under controlled env overrides.')
    parser.add_argument('--protocols', default='kfold,repeated,nested', help='Comma-separated protocols: kfold,repeated,nested')
    parser.add_argument('--selected-models', default='HistGB,RandomForest', help='SELECTED_MODELS env value passed to main.py')
    parser.add_argument('--cv-repeats', default='3', help='Repeat count used when protocol includes repeated.')
    parser.add_argument('--nested-inner-folds', default='3', help='Inner folds used when protocol includes nested.')
    parser.add_argument('--nested-rf-grid', default='light', help='Nested RF grid profile.')
    parser.add_argument('--enable-shap', action='store_true', help='Enable SHAP during comparison runs.')
    parser.add_argument('--enable-perm-importance', action='store_true', help='Enable permutation importance during comparison runs.')
    parser.add_argument('--enable-eval-plots', action='store_true', help='Enable evaluation plot generation.')
    parser.add_argument('--dry-run', action='store_true', help='Print planned commands/env without executing.')
    return parser.parse_args(argv)


def run_once(run_tag: str, env: dict, *, dry_run: bool = False) -> None:
    e = os.environ.copy()
    e.update(env)
    e['OUTPUT_ROOT_DIR'] = 'validation_compare'
    e['RUN_TAG'] = run_tag
    # 10-fold setting
    if run_tag == 'kfold':
        e['CV_MODE'] = 'kfold'
        e['CV_FOLDS'] = '10'
    elif run_tag == 'repeated':
        e['CV_MODE'] = 'repeated'
        e['CV_FOLDS'] = '10'
        e['CV_REPEATS'] = e.get('CV_REPEATS', '3')
    elif run_tag == 'nested':
        e['CV_MODE'] = 'nested'
        e['NESTED_OUTER_FOLDS'] = '10'
        # Use lighter inner CV to keep runtime reasonable
        e['NESTED_INNER_FOLDS'] = e.get('NESTED_INNER_FOLDS', '3')
        # Light RF grid for nested search to avoid long runtimes
        e['NESTED_RF_GRID'] = e.get('NESTED_RF_GRID', 'light')
    # Performance defaults if caller did not provide explicit overrides.
    e['PI_ONLY_BEST_MODEL'] = e.get('PI_ONLY_BEST_MODEL', 'true')
    e['PI_N_JOBS'] = e.get('PI_N_JOBS', '-1')
    cmd = [PY, '-X', 'faulthandler', os.path.join(ROOT, 'main.py')]
    LOGGER.info('RUN %s CV with OUTPUT_ROOT_DIR=validation_compare/%s', run_tag, run_tag)
    if dry_run:
        LOGGER.info('Dry-run command: %s', ' '.join(cmd))
        return
    subprocess.check_call(cmd, env=e, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    configure_logging(app_name='validation_compare')
    args = parse_args(argv)

    try:
        protocols = _parse_protocols(args.protocols)
    except ValueError as exc:
        LOGGER.error(str(exc))
        return 2

    base_env = {
        'CV_REPEATS': str(args.cv_repeats),
        'NESTED_INNER_FOLDS': str(args.nested_inner_folds),
        'NESTED_RF_GRID': str(args.nested_rf_grid),
        'PI_ONLY_BEST_MODEL': os.environ.get('PI_ONLY_BEST_MODEL', 'true'),
        'PI_N_JOBS': os.environ.get('PI_N_JOBS', '-1'),
        'SELECTED_MODELS': str(args.selected_models),
        'DO_SHAP': 'true' if args.enable_shap else 'false',
        'PERM_IMPORTANCE_ENABLED': 'true' if args.enable_perm_importance else 'false',
        'EVAL_PLOTS_ENABLED': 'true' if args.enable_eval_plots else 'false',
    }

    for protocol in protocols:
        try:
            run_once(protocol, base_env, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            LOGGER.error('Protocol %s failed with return code %s', protocol, exc.returncode)
            return exc.returncode

    LOGGER.info('Validation comparison pipeline completed for protocols: %s', ', '.join(protocols))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
