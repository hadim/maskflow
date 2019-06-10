from .system import get_system


def _get_win_folder_from_registry(csidl_name):
  # type: (str) -> str
  """
  This is a fallback technique at best. I'm not sure if using the
  registry for this guarantees us the correct answer for all CSIDL_*
  names.
  """
  import _winreg  # pylint: disable=import-error

  shell_folder_name = {
      "CSIDL_APPDATA": "AppData",
      "CSIDL_COMMON_APPDATA": "Common AppData",
      "CSIDL_LOCAL_APPDATA": "Local AppData",
  }[csidl_name]

  key = _winreg.OpenKey(
      _winreg.HKEY_CURRENT_USER,
      r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
  )
  directory, _ = _winreg.QueryValueEx(key, shell_folder_name)
  return directory


def _get_win_folder_with_ctypes(csidl_name):
  import ctypes

  csidl_const = {
      "CSIDL_APPDATA": 26,
      "CSIDL_COMMON_APPDATA": 35,
      "CSIDL_LOCAL_APPDATA": 28,
  }[csidl_name]

  buf = ctypes.create_unicode_buffer(1024)
  ctypes.windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)

  # Downgrade to short path name if have highbit chars. See
  # <http://bugs.activestate.com/show_bug.cgi?id=85099>.
  has_high_char = False
  for c in buf:
    if ord(c) > 255:
      has_high_char = True
      break
  if has_high_char:
    buf2 = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
      buf = buf2

  return buf.value


def get_win_folder(csidl_name):

  if get_system() == "win32":
    try:
      return _get_win_folder_with_ctypes(csidl_name)
    except ImportError:
      return _get_win_folder_from_registry(csidl_name)
  else:
    return None
