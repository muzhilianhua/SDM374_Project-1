"""Background jobs: launch real sdm commands, follow their status and logs, cancel them."""
import sys
import time

from sdm.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "app"))
import jobs  # noqa: E402


def wait(job, timeout=120):
    deadline = time.monotonic() + timeout
    while job.status == jobs.RUNNING and time.monotonic() < deadline:
        time.sleep(0.2)
    return job.status


def test_train_job_succeeds_and_links_its_run(tmp_path):
    job = jobs.start(["train", "-c", "experiments/baseline.toml"], tmp_path)
    assert wait(job) == jobs.SUCCEEDED, job.log()
    assert job.exit_code == 0
    assert (job.run_path / "metrics.json").exists()
    assert job.run_path.parent == tmp_path


def test_failing_command_is_reported(tmp_path):
    job = jobs.start(["train", "-c", "experiments/missing.toml"], tmp_path)
    assert wait(job) == jobs.FAILED
    assert "error:" in job.log()


def test_cancel_stops_a_running_job(tmp_path):
    job = jobs.start(["sleep"], tmp_path, program=[sys.executable, "-c", "import time; time.sleep(60)"])
    time.sleep(1)
    assert job.status == jobs.RUNNING
    jobs.cancel(job)
    assert wait(job, timeout=10) == jobs.CANCELLED


def test_jobs_are_listed_newest_first_and_hidden_from_runs(tmp_path):
    first = jobs.start(["list"], tmp_path)
    wait(first)
    time.sleep(1.1)
    second = jobs.start(["list"], tmp_path)
    wait(second)
    assert [job.id for job in jobs.list_jobs(tmp_path)] == [second.id, first.id]
    from sdm.runs import list_runs
    assert list_runs(tmp_path) == []
