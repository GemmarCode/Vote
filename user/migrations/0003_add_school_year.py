from django.db import migrations, models
from django.utils import timezone

def get_default_school_year():
    current_year = timezone.now().year
    return f"{current_year}-{current_year + 1}"

class Migration(migrations.Migration):

    dependencies = [
        ('user', '0002_verificationcode'),
    ]

    operations = [
        migrations.AddField(
            model_name='candidate',
            name='school_year',
            field=models.CharField(
                max_length=9,
                default=get_default_school_year,
                help_text='Format: YYYY-YYYY'
            ),
        ),
        migrations.AlterUniqueTogether(
            name='candidate',
            unique_together={('user_profile', 'school_year')},
        ),
    ] 