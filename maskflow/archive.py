from pathlib import Path
import sys
import os
import tqdm
import zipfile
import tempfile
import urllib
import shutil


__all__ = ['open_archive', 'user_data_dir', 'download_zip']


class _TqdmUpTo(tqdm.tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Args:
            b: int, optional
                Number of blocks transferred so far [default: 1].
            bsize: int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize: int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

        
def download_zip(zip_url, extract_folder_path, progressbar=True):
    """Download a ZIP file from an URL and extract to a given local folder.
    
    Args:
        zip_url: The URL to the ZIP file as a str.
        extract_folder_path: The path to the local folder for the extraction.
    """
    temp_path = tempfile.mktemp(suffix=".zip")

    with _TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, disable=not progressbar) as t:
        urllib.request.urlretrieve(zip_url, filename=temp_path, reporthook=t.update_to, data=None)

    with zipfile.ZipFile(temp_path) as zf:
        zf.extractall(extract_folder_path)

    os.remove(temp_path)
    

def _get_system():
    if sys.platform.startswith('java'):
        import platform
        os_name = platform.java_ver()[3][0]
        if os_name.startswith('Windows'): # "Windows XP", "Windows 7", etc.
            system = 'win32'
        elif os_name.startswith('Mac'): # "Mac OS X", etc.
            system = 'darwin'
        else: # "Linux", "SunOS", "FreeBSD", etc.
            # Setting this to "linux2" is not ideal, but only Windows or Mac
            # are actually checked for and the rest of the module expects
            # *sys.platform* style strings.
            system = 'linux2'
    else:
        system = sys.platform


# Source:
# https://github.com/ActiveState/appdirs/blob/0a7725ce30fbe6b7a6992458b0ba55f998195a6f/appdirs.py
def user_data_dir(appname=None, appauthor=None, version=None, roaming=False):
    r"""Return full path to the user-specific data dir for this application.
        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "roaming" (boolean, default False) can be set True to use the Windows
            roaming appdata directory. That means that for users on a Windows
            network setup for roaming profiles, this user data will be
            sync'd on login. See
            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>
            for a discussion of issues.
    Typical user data directories are:
        Mac OS X:               ~/Library/Application Support/<AppName>
        Unix:                   ~/.local/share/<AppName>    # or in $XDG_DATA_HOME, if defined
        Win XP (not roaming):   C:\Documents and Settings\<username>\Application Data\<AppAuthor>\<AppName>
        Win XP (roaming):       C:\Documents and Settings\<username>\Local Settings\Application Data\<AppAuthor>\<AppName>
        Win 7  (not roaming):   C:\Users\<username>\AppData\Local\<AppAuthor>\<AppName>
        Win 7  (roaming):       C:\Users\<username>\AppData\Roaming\<AppAuthor>\<AppName>
    For Unix, we follow the XDG spec and support $XDG_DATA_HOME.
    That means, by default "~/.local/share/<AppName>".
    """
    
    system = _get_system()
    
    if system == "win32":
        if appauthor is None:
            appauthor = appname
        const = roaming and "CSIDL_APPDATA" or "CSIDL_LOCAL_APPDATA"
        path = os.path.normpath(_get_win_folder(const))
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


def open_archive(archive_location, appname, erase=False, progressbar=False):
    """Exctract ZIP archive into the system data directory.
    
    Args:
        archive_location: `Path` or string, can be either a local file or an URL.
        appname: string, name to use for the data directory.
        erase: bool, erase archive folder.
        progressbar: bool, during download.
        
    Returns:
        A `Path` object to the folder.
    """
    
    data_dir = user_data_dir(appname='maskflow')
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
