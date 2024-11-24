from django.db import models

# Create your models here.
from django import forms

class MobilePriceForm(forms.Form):
    battery_power = forms.IntegerField(label="Battery Power (mAh)", min_value=0)
    bluetooth = forms.BooleanField(label="Has Bluetooth", required=False)
    clock_speed = forms.FloatField(label="Clock Speed (GHz)", min_value=0.0)
    dual_sim = forms.BooleanField(label="Has Dual SIM", required=False)
    front_camera = forms.IntegerField(label="Front Camera (MP)", min_value=0)
    four_g = forms.BooleanField(label="Has 4G", required=False)
    internal_memory = forms.IntegerField(label="Internal Memory (GB)", min_value=0)
    mobile_depth = forms.FloatField(label="Mobile Depth (cm)", min_value=0.0)
    weight = forms.IntegerField(label="Weight (g)", min_value=0)
    cores = forms.IntegerField(label="Processor Cores", min_value=1)
    primary_camera = forms.IntegerField(label="Primary Camera (MP)", min_value=0)
    pixel_height = forms.IntegerField(label="Pixel Height", min_value=0)
    pixel_width = forms.IntegerField(label="Pixel Width", min_value=0)
    ram = forms.IntegerField(label="RAM (MB)", min_value=0)
    screen_height = forms.IntegerField(label="Screen Height (cm)", min_value=0)
    screen_width = forms.IntegerField(label="Screen Width (cm)", min_value=0)
    battery_life = forms.IntegerField(label="Battery Life (hours)", min_value=0)
    three_g = forms.BooleanField(label="Has 3G", required=False)
    touch_screen = forms.BooleanField(label="Has Touch Screen", required=False)
    wifi = forms.BooleanField(label="Has WiFi", required=False)
