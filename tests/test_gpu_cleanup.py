from pySL_gpu_cleanup import GpuProcess, is_owned_pysl_process


def test_owned_pysl_batch_process_is_candidate():
    process = GpuProcess(
        pid=12345,
        process_name="python",
        used_memory="1024",
        cmdline="/usr/bin/python pySLbatch.py --config demo",
        owned_by_user=True,
    )

    assert is_owned_pysl_process(process)


def test_other_user_pysl_process_is_not_candidate():
    process = GpuProcess(
        pid=12345,
        process_name="python",
        used_memory="1024",
        cmdline="/usr/bin/python pySLbatch.py --config demo",
        owned_by_user=False,
    )

    assert not is_owned_pysl_process(process)


def test_cleanup_script_is_not_candidate():
    process = GpuProcess(
        pid=12345,
        process_name="python",
        used_memory="10",
        cmdline="/usr/bin/python pySL_gpu_cleanup.py --kill-owned-pysl",
        owned_by_user=True,
    )

    assert not is_owned_pysl_process(process)


def test_registered_owned_worker_is_candidate_without_pysl_cmdline_marker():
    process = GpuProcess(
        pid=12345,
        process_name="python",
        used_memory="1024",
        cmdline="/usr/bin/python -c from multiprocessing.spawn import spawn_main",
        owned_by_user=True,
        registered_pysl=True,
    )

    assert is_owned_pysl_process(process)
