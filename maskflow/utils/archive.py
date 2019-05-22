
import shutil
import os
from pathlib import Path
import zipfile

from .download import download_zip
from .system import get_system
from .windows import get_win_folder


def user_data_dir(appname=None, appauthor=None, version=None, roaming=False):
    """Return full path to the user-specific data dir for this application.

    Args:
        appname: `str`, the name of the application. If None, just the system
            directory is returned.
        appauthor: `str`, (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        version: `str`, is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        roaming: `bool`, (default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.

    Returns:
        A `Path` object to the directory.

    Typical user data directories are:
        Mac OS X: `~/Library/Application Support/<AppName>`
        Unix: `~/.local/share/<AppName>
        Windows (not roaming): `C:\\Users\\<username>\\AppData\\Local\\<AppAuthor>\\<AppName>`
        Windows (roaming): `C:\\Users\\<username>\\AppData\\Roaming\\<AppAuthor>\\<AppName>`
    """

    system = get_system()

    if system == "win32":
        if appauthor is None:
            appauthor = appname
        const = roaming and "CSIDL_APPDATA" or "CSIDL_LOCAL_APPDATA"
        path = os.path.normpath(get_win_folder(const))
        if appname:
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                path = os.path.join(path, appname)
    elif system == 'darwin':
        path = os.path.expanduser('~/Library/Application Support/')
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv('XDG_DATA_HOME', os.path.expanduser("~/.local/share"))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return Path(path)


def open_archive(archive_location, appname="maskflow", erase=False, progressbar=False):
    """Extract ZIP archive into the system data directory. Download it if needed.

    Args:
        archive_location: `Path` or string, can be either a local file or an URL.
        appname: `str`, name to use for the data directory.
        erase: `bool`, erase archive folder.
        progressbar: `bool`, during download.

    Returns:
        A `Path` object to the folder.
    """

    data_dir = user_data_dir(appname=appname)
    data_dir.mkdir(exist_ok=True, parents=True)

    basename = Path(archive_location).stem
    archive_path = data_dir / basename

    if archive_path.is_dir() and erase:
        shutil.rmtree(archive_path)

    if archive_path.is_dir():
        return archive_path

    if Path(archive_location).is_file():
        with zipfile.ZipFile(archive_location) as zf:
            zf.extractall(archive_path)
    else:
        download_zip(archive_location, archive_path, progressbar=progressbar)

    return archive_path
