import signal

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.delegated as food
import fiftyone.operators.delegated_executors.continual as fodec


def main():
    do_svc = food.DelegatedOperationService()
    orch_svc = foo.orchestrator.OrchestratorService()
    log_path = fo.config.delegated_operation_log_path
    kwargs = {
        "execution_interval": 10,
        "log_directory_path": log_path,
    }
    executor = fodec.ContinualExecutor(
        do_svc,
        orch_svc,
        **kwargs,
    )
    signal.signal(signal.SIGTERM, executor.signal_handler)
    signal.signal(signal.SIGINT, executor.signal_handler)
    executor.start()


if __name__ == '__main__':
    main()