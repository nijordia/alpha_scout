"""
A simple implementation of the removed imghdr module
"""

def what(file, h=None):
    """Determine the type of image contained in a file or byte stream."""
    if h is None:
        if isinstance(file, (str, bytes)):
            with open(file, 'rb') as f:
                h = f.read(32)
        else:
            location = file.tell()
            h = file.read(32)
            file.seek(location)
    
    # JPG
    if h[0:2] == b'\xff\xd8':
        return 'jpeg'
    # PNG
    if h[0:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    # GIF
    if h[0:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    # TIFF
    if h[0:2] in (b'MM', b'II'):
        return 'tiff'
    # BMP
    if h[0:2] == b'BM':
        return 'bmp'
    
    return None