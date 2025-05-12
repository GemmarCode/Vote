from .models import ChairmanAccount, CommitteeAccount

def role_flags(request):
    is_chairman = False
    is_committee = False
    if request.user.is_authenticated:
        is_chairman = ChairmanAccount.objects.filter(user=request.user).exists()
        is_committee = CommitteeAccount.objects.filter(user=request.user).exists()
    return {
        'is_chairman': is_chairman,
        'is_committee': is_committee,
    } 