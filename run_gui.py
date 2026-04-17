from utils.logger import configure_logging


configure_logging("ml_trainer.gui")

from interface.qt_app import run_app

if __name__ == "__main__":
    run_app()
