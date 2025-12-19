from django.db import models

# Create your models here.
class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(unique=True, max_length=100)
    email = models.CharField(unique=True, max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'UserRegistrations'

from django.db import models

class InputData(models.Model):
    # Choices for TypeEtab (Public/Private)
    TypeEtab_choices = (
        ('Public', 'Public'),
        ('Private', 'Private'),
    )

    # Choices for Genre (Male/Female)
    Genre_choices = (
        ('Male', 'Male'),
        ('Female', 'Female'),
    )

    # Choices for RetardSco (0, 1, 2 years)
    RetardSco_choices = (
        ('0', '0 years'),
        ('1', '1 year'),
        ('2', '2 years'),
    )

    # Choices for Handicap (Yes/No)
    Handicap_choices = (
        ('Yes', 'Yes'),
        ('No', 'No'),
    )

    # Choices for Fee_reimbursement (Yes/No)
    Feereimbursement_choices = (
        ('Yes', 'Yes'),
        ('No', 'No'),
    )

    # Choices for Result (Continue/Drop/Pass/Fail etc.)
    Result_choices = (
        ('Continue', 'Continue'),
        ('Drop', 'Drop'),
        ('Pass', 'Pass'),
        ('Fail', 'Fail'),
    )

    # Fields
    TypeEtab = models.CharField(max_length=10)
    Genre = models.CharField(max_length=6)
    Niveau = models.CharField(max_length=50)
    RetardSco = models.CharField(max_length=1)  # max_length=1 to store '0', '1', or '2'
    Provenance = models.CharField(max_length=50)
    Fee_reimbursement = models.CharField(max_length=3)
    Moy = models.FloatField()
    Result = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.Niveau} - {self.Provenance} - {self.Genre} - {self.Result}"
