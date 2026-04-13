import subprocess

def get_git_info():

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()

    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    ).decode().strip()

    return{
        "git_commit": commit,
        "git_branch": branch
    }

def is_dirty():
    status = subprocess.check_output(
        ["git", "status", "--porcelain"]
    ).decode().strip()
    return len(status) > 0