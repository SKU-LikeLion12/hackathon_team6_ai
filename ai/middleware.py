from django.http import HttpResponse
from io import BytesIO

class ChunkedUploadMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.META.get('HTTP_TRANSFER_ENCODING') == 'chunked':
            content = b''
            while True:
                chunk = request.read(4096)
                if not chunk:
                    break
                content += chunk
            request._body = content
            request.META['CONTENT_LENGTH'] = len(content)
        return self.get_response(request)