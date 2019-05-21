$PROJECT = 'maskflow'

$ACTIVITIES = ['version_bump', 'tag', 'push_tag', 'pypi', 'ghrelease']

$VERSION_BUMP_PATTERNS = [('maskflow/_version.py', '__version__\s*=.*', "__version__ = '$VERSION'"),
                          ('setup.py', 'version\s*=.*,', "version='$VERSION',")]

$PUSH_TAG_REMOTE = 'git@github.com:hadim/maskflow.git'
$GITHUB_ORG = 'hadim'
$GITHUB_REPO = 'maskflow'
