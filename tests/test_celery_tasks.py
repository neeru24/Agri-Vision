import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "celery_tasks.py"


def _function_named(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} was not found")


def test_celery_image_task_accepts_batch_image_metadata():
    tree = ast.parse(SOURCE.read_text())
    task = _function_named(tree, "analyze_image_task")

    arg_names = [arg.arg for arg in task.args.args]

    assert arg_names[:6] == [
        "self",
        "job_id",
        "result_id",
        "image_b64",
        "image_name",
        "image_index",
    ]


def test_process_batch_job_passes_image_metadata_to_celery_task():
    tree = ast.parse(SOURCE.read_text())
    process_batch_job = _function_named(tree, "process_batch_job")

    signature_calls = [
        node
        for node in ast.walk(process_batch_job)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "s"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "analyze_image_task"
    ]

    assert signature_calls, "process_batch_job must create analyze_image_task signatures"

    task_args = signature_calls[0].args
    assert len(task_args) == 5
    assert isinstance(task_args[3], ast.Name)
    assert task_args[3].id == "image_name"
    assert isinstance(task_args[4], ast.Name)
    assert task_args[4].id == "idx"
