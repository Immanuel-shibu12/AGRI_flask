"""
Core scheduling logic: turns (sowing_date + crop_stages) into a concrete
calendar of dates, and classifies each task as done / today / upcoming / overdue.
This is what would drive push notifications in a production mobile app —
here it drives the in-app "notification feed" for the demo.
"""

from datetime import date, datetime, timedelta


def _to_date(d):
    """
    Convert various inputs to date object.
    Handles: date objects, date strings, None, invalid inputs.
    Falls back gracefully to today's date if parsing fails.
    """
    # If already a date object, return it
    if isinstance(d, date):
        return d
    
    # If not a string, return today's date
    if not isinstance(d, str):
        return date.today()
    
    # Don't try to parse language codes or short strings like 'en', 'hi', 'es'
    # Min valid date string is "2024-01-01" (10 chars)
    if len(d.strip()) < 8:
        return date.today()
    
    # Try multiple date formats in order of likelihood
    date_formats = [
        "%Y-%m-%d",      # ISO format: 2024-06-15
        "%d-%m-%Y",      # European: 15-06-2024
        "%d/%m/%Y",      # European slash: 15/06/2024
        "%Y/%m/%d",      # ISO slash: 2024/06/15
        "%d-%b-%Y",      # Short month: 15-Jun-2024
        "%B %d, %Y",     # Full month: June 15, 2024
        "%b %d, %Y",     # Short month: Jun 15, 2024
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(d.strip(), fmt).date()
        except ValueError:
            continue
    
    # If all formats fail, log warning and use today's date
    print(f"⚠️ WARNING: Could not parse date '{d}' in any known format. Using today's date.")
    return date.today()


def build_calendar(sowing_date, stages):
    """
    Build a complete calendar with status tracking.
    
    Args:
        sowing_date: Date when crop was/will be sown (str, date, or None)
        stages: List of dicts from database with 'day_offset', 'stage_name', etc.
    
    Returns:
        List of dicts with added fields: due_date, days_from_today, status
        status values: "done", "today", "upcoming", "overdue"
    """
    sowing_date = _to_date(sowing_date)
    today = date.today()
    calendar = []

    for stage in stages:
        due_date = sowing_date + timedelta(days=stage["day_offset"])
        days_from_today = (due_date - today).days

        # Classify task status
        if days_from_today > 0:
            status = "upcoming"
        elif days_from_today == 0:
            status = "today"
        elif days_from_today < 0 and due_date >= today - timedelta(days=2):
            # Small grace window so a task doesn't vanish instantly at midnight
            status = "overdue"
        else:
            status = "done"

        # Build entry with added fields
        entry = dict(stage)
        entry["due_date"] = due_date
        entry["days_from_today"] = days_from_today
        entry["status"] = status
        calendar.append(entry)

    return calendar


def get_notifications(calendar, window_days=3):
    """
    Get tasks that need farmer attention right now.
    Returns: Tasks due today, overdue, or upcoming within window_days.
    
    This is the 'automatic schedule reminder' feed that would drive
    push notifications in a mobile app.
    """
    return [
        e for e in calendar
        if e["status"] in ("today", "overdue")
        or (e["status"] == "upcoming" and e["days_from_today"] <= window_days)
    ]


def get_harvest_info(calendar):
    """
    Extract harvest date info from calendar.
    Returns: Dict with harvest_date, days_left, stage_name
    or None if no harvest stages found.
    """
    harvest_entries = [e for e in calendar if e.get("category") == "harvest"]
    if not harvest_entries:
        return None
    
    # Use the last harvest-tagged stage (main harvest date)
    harvest = sorted(harvest_entries, key=lambda e: e["day_offset"])[-1]
    days_left = harvest["days_from_today"]
    
    return {
        "harvest_date": harvest["due_date"],
        "days_left": days_left,
        "stage_name": harvest["stage_name"],
    }


def apply_weather_override(calendar, weather_flag):
    """
    Apply weather advisories to weather-sensitive tasks.
    
    weather_flag: dict like {
        "rain_today": True,
        "high_wind_today": False,
        "reason": "..."
    }
    
    Does not change underlying schedule dates — just flags advisory text,
    since actual replanning is a farmer decision.
    
    Returns: Updated calendar with "weather_warning" field added to entries.
    """
    for entry in calendar:
        entry["weather_warning"] = None
        
        # Only add warnings for tasks that are happening now
        if entry.get("weather_sensitive") and entry["status"] in ("today", "overdue"):
            
            # Pesticide spraying weather warnings
            if entry["category"] in ("pesticide", "spray"):
                if weather_flag.get("rain_today"):
                    entry["weather_warning"] = (
                        "🌧️ Rain expected/occurring today — delay pesticide spraying "
                        "to avoid wash-off and ensure better coverage."
                    )
                elif weather_flag.get("high_wind_today"):
                    entry["weather_warning"] = (
                        "💨 High wind today — delay spraying to avoid drift and "
                        "ensure pesticide stays on target."
                    )
            
            # Irrigation weather warnings
            elif entry["category"] == "irrigation":
                if weather_flag.get("rain_today"):
                    entry["weather_warning"] = (
                        "🌧️ Rain expected/occurring today — you can likely skip "
                        "irrigation. Save water and monitor soil moisture tomorrow."
                    )
    
    return calendar


def get_stage_progress(calendar):
    """
    Calculate overall crop progress as a percentage.
    
    Returns: {
        "current_stage": "stage_name",
        "progress_percent": 0-100,
        "days_elapsed": N,
        "days_remaining": N,
        "total_cycle_days": N
    }
    """
    if not calendar:
        return None
    
    total_days = sum(s["day_offset"] for s in calendar)
    current_stage = next((s for s in calendar if s["status"] == "today"), None) or calendar[0]
    days_elapsed = current_stage["day_offset"]
    progress = int((days_elapsed / total_days) * 100) if total_days > 0 else 0
    
    return {
        "current_stage": current_stage.get("stage_name", "Unknown"),
        "progress_percent": min(100, progress),
        "days_elapsed": days_elapsed,
        "days_remaining": total_days - days_elapsed,
        "total_cycle_days": total_days,
    }


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Example crop stages for testing
    test_stages = [
        {"day_offset": 0, "stage_name": "Sowing", "category": "sowing", "weather_sensitive": False},
        {"day_offset": 14, "stage_name": "Germination", "category": "growth", "weather_sensitive": False},
        {"day_offset": 45, "stage_name": "First Spray", "category": "pesticide", "weather_sensitive": True},
        {"day_offset": 75, "stage_name": "Second Spray", "category": "pesticide", "weather_sensitive": True},
        {"day_offset": 120, "stage_name": "Harvest", "category": "harvest", "weather_sensitive": False},
    ]
    
    # Test with various date inputs
    print("=" * 60)
    print("Testing _to_date() function:")
    print("=" * 60)
    
    test_inputs = [
        "2024-06-15",      # Valid ISO format
        "15-06-2024",      # Valid European format
        "en",              # Invalid: language code (should return today)
        "hi",              # Invalid: language code
        None,              # None: should return today
        123,               # Invalid type: should return today
        date.today(),      # Valid date object
    ]
    
    for inp in test_inputs:
        result = _to_date(inp)
        print(f"  _to_date({inp!r:20}) → {result}")
    
    print("\n" + "=" * 60)
    print("Testing build_calendar():")
    print("=" * 60)
    
    # Build calendar starting from today
    calendar = build_calendar(date.today(), test_stages)
    
    for entry in calendar:
        status_emoji = {
            "done": "✅",
            "today": "📌",
            "upcoming": "⏰",
            "overdue": "⚠️"
        }
        emoji = status_emoji.get(entry["status"], "❓")
        print(f"{emoji} {entry['stage_name']:20} | Due: {entry['due_date']} | {entry['days_from_today']:+3}d | {entry['status']}")
    
    print("\n" + "=" * 60)
    print("Notifications (next 3 days):")
    print("=" * 60)
    
    notifs = get_notifications(calendar, window_days=3)
    if notifs:
        for n in notifs:
            print(f"  • {n['stage_name']} - {n['status'].upper()}")
    else:
        print("  No notifications")
    
    print("\n" + "=" * 60)
    print("Harvest Info:")
    print("=" * 60)
    
    harvest = get_harvest_info(calendar)
    if harvest:
        print(f"  Harvest: {harvest['stage_name']}")
        print(f"  Date: {harvest['harvest_date']}")
        print(f"  Days left: {harvest['days_left']}")
    
    print("\n" + "=" * 60)
    print("Crop Progress:")
    print("=" * 60)
    
    progress = get_stage_progress(calendar)
    if progress:
        print(f"  Current: {progress['current_stage']}")
        print(f"  Progress: {progress['progress_percent']}%")
        print(f"  Days: {progress['days_elapsed']}/{progress['total_cycle_days']}")