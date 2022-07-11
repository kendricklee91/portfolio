# method
http_request_method = {
        'OPTIONS ' : 180,
        'GET ' : 181,
        'HEAD ' : 182,
        'POST ' : 183,
        'PUT ' : 184,
        'DELETE ' : 185,
        'TRACE ' : 186,
        'CONNECT ' : 187
    }

http_request_ver = {
    'HTTP/1.1' : 189,
    'HTTP/1.0' : 189
}

# header
http_header_field = {
        # request header
        'Accept: ' : 200,
        'Accept-Charset: ' : 200,
        'Accept-Encoding: ' : 200,
        'Accept-Language: ' : 200,
        'From: ' : 200,
        'Host: ' : 200,
        'If-Match: ' : 200,
        'If-Modified-Since: ' : 200,
        'If-None-Match: ' : 200,
        'If-Range: ' : 200,
        'If-Unmodified-Since: ' : 200,
        'Max-Forwards: ' : 200,
        'Proxy-Authorization: ' : 200,
        'Range: ' : 200,
        'Referer: ' : 200,
        'User-Agent: ' : 200,
        # response header
        'Accept-Ranges: ' : 201,
        'Age: ' : 201,
        'Authorization: ' : 201,
        'Location: ' : 201,
        'Proxy-Authenticate: ' : 201,
        'Retry-After: ' : 201,
        'Server: ' : 201,
        'Vary: ' : 201,
        'Warning: ' : 201,
        'WWW-Authenticate: ' : 201,
        # enentity header
        'Allow: ' : 202,
        'Content-Encoding: ' : 202,
        'Content-Language: ' : 202,
        'Content-Length: ' : 202,
        'Content-Location: ' : 202,
        'Content-MD5: ' : 202,
        'Content-Range: ' : 202,
        'Content-Type: ' : 202,
        'ETag: ' : 202,
        'Expires: ' : 202,
        'Last-Modified: ' : 202,
        # etc header
        'Cache-Control: ' : 203,
        'Connection: ' : 203,
        'Date: ' : 203,
        'Expect: ' : 203,
        'Pragma: ' : 203,
        'TE: ' : 203,
        'Trailer: ' : 203,
        'Transfer-Encoding: ' : 203,
        'Upgrade: ' : 203,
        'Via: ' : 203
    }

# response code
http_response_status_code = {
        r"HTTP/\d\.\d\s1\d{2}" : 190,
        r"HTTP/\d\.\d\s2\d{2}" : 191,
        r"HTTP/\d\.\d\s3\d{2}" : 192,
        r"HTTP/\d\.\d\s4\d{2}" : 193,
        r"HTTP/\d\.\d\s5\d{2}" : 194
    }