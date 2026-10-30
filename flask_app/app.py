"""
AgriSchedule — Complete Flask Application
Full-featured smart crop calendar, weather advisory, visual pest diagnosis,
harvest countdown, government schemes directory, admin operations, and Mim AI Assistant.
"""

from datetime import datetime, date, timedelta
import io
import os

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from PIL import Image
import numpy as np

import database as db
import schedule_engine
import weather_utils
import pest_guidance
import schemes_data
import mim_chatbot

app = Flask(__name__)
app.secret_key = "agrischedule-secret-key-2026"
db.init_db()


def _get_farmer_or_default(farmer_id=None):
    farmers = db.get_all_farmers()
    if not farmers:
        return None, []
    if farmer_id:
        for f in farmers:
            if str(f["id"]) == str(farmer_id):
                return f, farmers
    return farmers[0], farmers


# ---------------------------------------------------------------------
# 0. Welcome / Splash Page
# ---------------------------------------------------------------------
@app.route("/", methods=["GET"])
@app.route("/welcome", methods=["GET"])
def welcome_page():
    farmers = db.get_all_farmers()
    total_farmers = len(farmers)
    total_locations = len({f["location"] for f in farmers if f.get("location")})
    # If farmers exist, continue can go to schedule or register
    continue_url = url_for("register_page")
    return render_template(
        "welcome.html",
        active_page="welcome",
        continue_url=continue_url,
        total_farmers=total_farmers,
        total_locations=total_locations,
    )


# ---------------------------------------------------------------------
# 1. Farmer Registration
# ---------------------------------------------------------------------
@app.route("/register", methods=["GET"])
def register_page():
    farmers = db.get_all_farmers()
    crops = db.get_crops()
    total_farmers = len(farmers)
    total_acres = round(sum(f["field_area"] or 0 for f in farmers), 1)
    crop_types = len({f["crop"] for f in farmers if f.get("crop")})
    locations = len({f["location"] for f in farmers if f.get("location")})

    current_farmer = farmers[0] if farmers else None

    return render_template(
        "register.html",
        active_page="register",
        current_farmer=current_farmer,
        crops=crops,
        farmers=farmers,
        total_farmers=total_farmers,
        total_acres=total_acres,
        crop_types=crop_types,
        locations=locations,
    )


@app.route("/varieties/<crop>")
def varieties_for_crop(crop):
    return jsonify(db.get_varieties(crop))


@app.route("/register", methods=["POST"])
def submit_registration():
    name = request.form.get("name", "").strip()
    phone = request.form.get("phone", "").strip() or None
    location = request.form.get("location", "").strip()
    crop = request.form.get("crop")
    variety = request.form.get("variety")
    field_area = float(request.form.get("field_area") or 1.0)
    sowing_date = request.form.get("sowing_date")

    if name and location and sowing_date:
        new_id = db.add_farmer(name, phone, location, None, None, crop, variety, field_area, sowing_date, "en")
        flash(f"Farmer '{name}' registered successfully! Personalized calendar generated.", "success")
        return redirect(url_for("schedule_page", farmer_id=new_id))

    flash("Please fill in all required fields (Name, Location, Sowing Date).", "error")
    return redirect(url_for("register_page"))


@app.route("/farmer/delete/<int:farmer_id>", methods=["POST"])
def delete_farmer_route(farmer_id):
    farmer = db.get_farmer(farmer_id)
    if farmer:
        db.delete_farmer(farmer_id)
        flash(f"Farmer profile for '{farmer['name']}' was removed.", "info")
    return redirect(url_for("register_page"))


# ---------------------------------------------------------------------
# 2. My Crop Schedule
# ---------------------------------------------------------------------
@app.route("/schedule")
@app.route("/schedule/<int:farmer_id>")
def schedule_page(farmer_id=None):
    current_farmer, farmers = _get_farmer_or_default(farmer_id)
    if not current_farmer:
        return render_template("schedule.html", active_page="schedule", current_farmer=None, farmers=[])

    stages = db.get_schedule(current_farmer["crop"], current_farmer["variety"])
    calendar = schedule_engine.build_calendar(current_farmer["sowing_date"], stages)

    # Weather checks for overrides
    geo = weather_utils.geocode_location(current_farmer["location"])
    weather_flag = {"available": False}
    if geo:
        forecast = weather_utils.get_forecast(geo["latitude"], geo["longitude"], days=3)
        if forecast:
            weather_flag = weather_utils.get_today_flags(forecast)
            calendar = schedule_engine.apply_weather_override(calendar, weather_flag)

    notifications = schedule_engine.get_notifications(calendar, window_days=3)
    progress = schedule_engine.get_stage_progress(calendar)

    return render_template(
        "schedule.html",
        active_page="schedule",
        current_farmer=current_farmer,
        farmers=farmers,
        calendar=calendar,
        notifications=notifications,
        progress=progress,
        weather_flag=weather_flag,
    )


# ---------------------------------------------------------------------
# 3. Weather Advisory
# ---------------------------------------------------------------------
@app.route("/weather")
@app.route("/weather/<int:farmer_id>")
def weather_page(farmer_id=None):
    current_farmer, farmers = _get_farmer_or_default(farmer_id)
    location_name = request.args.get("location")
    if not location_name and current_farmer:
        location_name = current_farmer["location"]
    if not location_name:
        location_name = "Thrissur, Kerala"

    geo = weather_utils.geocode_location(location_name)
    forecast = None
    today_flags = {"available": False}
    daily_list = []

    if geo:
        forecast = weather_utils.get_forecast(geo["latitude"], geo["longitude"], days=7)
        if forecast and "time" in forecast:
            today_flags = weather_utils.get_today_flags(forecast)
            today_date = date.today()
            for i, time_str in enumerate(forecast["time"]):
                try:
                    d = datetime.strptime(time_str, "%Y-%m-%d").date()
                except Exception:
                    d = today_date + timedelta(days=i)
                daily_list.append({
                    "date": d,
                    "rain": forecast["precipitation_sum"][i] or 0.0,
                    "wind": forecast["windspeed_10m_max"][i] or 0.0,
                    "temp_max": forecast["temperature_2m_max"][i] or 0.0,
                    "temp_min": forecast["temperature_2m_min"][i] or 0.0,
                })

    return render_template(
        "weather.html",
        active_page="weather",
        current_farmer=current_farmer,
        farmers=farmers,
        location_name=location_name,
        geo=geo,
        forecast=forecast,
        today_flags=today_flags,
        daily_list=daily_list,
    )


# ---------------------------------------------------------------------
# 4. Pest & Disease Guidance
# ---------------------------------------------------------------------
@app.route("/pest")
@app.route("/pest/<int:farmer_id>")
def pest_page(farmer_id=None):
    current_farmer, farmers = _get_farmer_or_default(farmer_id)
    recent_reports = db.get_all_pest_reports()
    return render_template(
        "pest.html",
        active_page="pest",
        current_farmer=current_farmer,
        farmers=farmers,
        recent_reports=recent_reports,
        diagnosis=None,
    )


@app.route("/pest/diagnose", methods=["POST"])
def diagnose_pest():
    farmer_id = request.form.get("farmer_id")
    sample_type = request.form.get("sample_type", "").strip()
    camera_data = request.form.get("camera_data", "").strip()
    current_farmer, farmers = _get_farmer_or_default(farmer_id)

    pil_img = None
    source_name = "Upload"

    if camera_data:
        try:
            import base64
            if "," in camera_data:
                camera_data = camera_data.split(",", 1)[1]
            img_bytes = base64.b64decode(camera_data)
            pil_img = Image.open(io.BytesIO(img_bytes))
            source_name = "Live Camera Scan"
        except Exception as e:
            flash(f"Could not decode camera scan: {e}", "error")
    elif sample_type:
        # Generate rich realistic demo image for 1-click presets
        from train_pest_model import generate_synthetic_leaf
        category_map = {
            "yellowing": "yellowing_leaves",
            "brown_spots": "brown_spots",
            "chewed_holes": "chewed_holes",
            "healthy": "healthy",
            "powdery_mildew": "powdery_mildew",
        }
        cat = category_map.get(sample_type, sample_type)
        pil_img = generate_synthetic_leaf(cat, seed=42)
        pil_img = pil_img.resize((380, 380), Image.Resampling.NEAREST)
        source_name = f"Sample ({sample_type})"
    elif "leaf_image" in request.files:
        file = request.files["leaf_image"]
        if file and file.filename:
            try:
                pil_img = Image.open(file.stream)
                source_name = file.filename
            except Exception as e:
                flash(f"Could not read uploaded image: {e}", "error")

    diagnosis = None
    if pil_img:
        diagnosis = pest_guidance.analyze_image(pil_img)
        # Log to DB
        fid = int(farmer_id) if farmer_id and str(farmer_id).isdigit() else (current_farmer["id"] if current_farmer else None)
        db.log_pest_report(fid, source_name, diagnosis["label"], diagnosis["confidence"])
        flash(f"Diagnosis completed via ML Model: {diagnosis['label']} ({int(diagnosis['confidence']*100)}% confidence)", "success")
    else:
        flash("Please upload a leaf photo, capture from live camera, or pick a sample preset.", "error")

    recent_reports = db.get_all_pest_reports()
    return render_template(
        "pest.html",
        active_page="pest",
        current_farmer=current_farmer,
        farmers=farmers,
        recent_reports=recent_reports,
        diagnosis=diagnosis,
    )


# ---------------------------------------------------------------------
# 5. Harvest Countdown
# ---------------------------------------------------------------------
@app.route("/harvest")
@app.route("/harvest/<int:farmer_id>")
def harvest_page(farmer_id=None):
    current_farmer, farmers = _get_farmer_or_default(farmer_id)
    harvest_info = None

    if current_farmer:
        stages = db.get_schedule(current_farmer["crop"], current_farmer["variety"])
        calendar = schedule_engine.build_calendar(current_farmer["sowing_date"], stages)
        harvest_info = schedule_engine.get_harvest_info(calendar)

    return render_template(
        "harvest.html",
        active_page="harvest",
        current_farmer=current_farmer,
        farmers=farmers,
        harvest_info=harvest_info,
    )


# ---------------------------------------------------------------------
# 6. Govt. Schemes & Subsidies
# ---------------------------------------------------------------------
@app.route("/schemes")
def schemes_page():
    current_farmer, farmers = _get_farmer_or_default()
    return render_template(
        "schemes.html",
        active_page="schemes",
        current_farmer=current_farmer,
        schemes=schemes_data.SCHEMES,
        disclaimer=schemes_data.DISCLAIMER,
    )


# ---------------------------------------------------------------------
# 7. Admin Dashboard
# ---------------------------------------------------------------------
@app.route("/admin")
def admin_page():
    current_farmer, farmers = _get_farmer_or_default()
    crop_stages = db.get_all_crop_stages()
    crops = db.get_crops()
    pest_reports = db.get_all_pest_reports()

    return render_template(
        "admin.html",
        active_page="admin",
        current_farmer=current_farmer,
        farmers=farmers,
        crop_stages=crop_stages,
        crops=crops,
        pest_reports=pest_reports,
    )


@app.route("/admin/add_stage", methods=["POST"])
def admin_add_stage():
    crop = request.form.get("crop", "").strip()
    variety = request.form.get("variety", "").strip()
    day_offset = int(request.form.get("day_offset") or 0)
    stage_name = request.form.get("stage_name", "").strip()
    category = request.form.get("category", "observation").strip()
    instructions = request.form.get("instructions", "").strip()
    weather_sensitive = 1 if request.form.get("weather_sensitive") else 0

    if crop and variety and stage_name and instructions:
        db.add_crop_stage(crop, variety, day_offset, stage_name, category, instructions, weather_sensitive)
        flash(f"Added stage '{stage_name}' to {crop} ({variety}) at day {day_offset}.", "success")
    else:
        flash("Please complete all required stage fields.", "error")

    return redirect(url_for("admin_page"))


@app.route("/admin/delete_stage/<int:stage_id>", methods=["POST"])
def admin_delete_stage(stage_id):
    db.delete_crop_stage(stage_id)
    flash(f"Crop stage #{stage_id} removed from master calendar.", "info")
    return redirect(url_for("admin_page"))


# ---------------------------------------------------------------------
# 8. Mim AI Assistant API
# ---------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def mim_chat_api():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    farmer_id = data.get("farmer_id")

    if not message:
        return jsonify({"reply": "Please enter a question for Mim."})

    farmer = db.get_farmer(farmer_id) if farmer_id else None
    context = ""
    if farmer:
        context = f"Farmer: {farmer['name']}, Crop: {farmer['crop']} ({farmer['variety']}), Location: {farmer['location']}, Sowing Date: {farmer['sowing_date']}, Field Area: {farmer['field_area']} acres."

    reply = mim_chatbot.ask_mim(message, context=context)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
