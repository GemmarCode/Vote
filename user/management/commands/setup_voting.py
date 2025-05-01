from django.core.management.base import BaseCommand
from django.utils import timezone
from user.models import VotingPhase
from admin_panel.models import ElectionSettings
import datetime

class Command(BaseCommand):
    help = 'Sets up voting phase for current time'

    def handle(self, *args, **options):
        # Create or update VotingPhase
        voting_phase, created = VotingPhase.objects.get_or_create(
            defaults={
                'phase': 'ONGOING',
                'start_date': timezone.now(),
                'end_date': timezone.now() + datetime.timedelta(hours=24)
            }
        )
        
        if not created:
            voting_phase.phase = 'ONGOING'
            voting_phase.start_date = timezone.now()
            voting_phase.end_date = timezone.now() + datetime.timedelta(hours=24)
            voting_phase.save()

        # Update ElectionSettings
        try:
            settings = ElectionSettings.objects.get(is_active=True)
            settings.voting_date = timezone.now().date()
            settings.voting_time_start = timezone.now().time()
            settings.voting_time_end = (timezone.now() + datetime.timedelta(hours=24)).time()
            settings.save()
            self.stdout.write(self.style.SUCCESS('Successfully set up voting phase'))
        except ElectionSettings.DoesNotExist:
            self.stdout.write(self.style.ERROR('No active election settings found')) 