from django.contrib.auth.decorators import user_passes_test
from django.core.exceptions import PermissionDenied

def superuser_required(view_func):
    def check_superuser(user):
        if not user.is_superuser:
            raise PermissionDenied
        return True
    
    decorated_view = user_passes_test(check_superuser)(view_func)
    return decorated_view 