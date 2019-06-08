import sys


def get_system():
  if sys.platform.startswith('java'):
    import platform
    os_name = platform.java_ver()[3][0]
    if os_name.startswith('Windows'):  # "Windows XP", "Windows 7", etc.
      system = 'win32'
    elif os_name.startswith('Mac'):  # "Mac OS X", etc.
      system = 'darwin'
    else:  # "Linux", "SunOS", "FreeBSD", etc.
      # Setting this to "linux2" is not ideal, but only Windows or Mac
      # are actually checked for and the rest of the module expects
      # *sys.platform* style strings.
      system = 'linux2'
  else:
    system = sys.platform

  return system
