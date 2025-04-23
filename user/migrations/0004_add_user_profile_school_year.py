from django.db import migrations, models
from django.utils import timezone

def get_default_school_year():
    current_year = timezone.now().year
    return f"{current_year}-{current_year + 1}"

class Migration(migrations.Migration):

    dependencies = [
        ('user', '0003_add_school_year'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='school_year',
            field=models.CharField(
                max_length=9,
                default=get_default_school_year,
                help_text='Format: YYYY-YYYY'
            ),
        ),
        migrations.AlterUniqueTogether(
            name='userprofile',
            unique_together={('student_number', 'school_year')},
        ),
    ] 