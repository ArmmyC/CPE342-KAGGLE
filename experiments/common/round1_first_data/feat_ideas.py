"""Feature ideas proposed by the LLM from the datacards, one function per idea.
Each takes the raw feature frame and returns only the new columns. feat_search.py scores them.
Trees already handle monotone transforms, so ideas focus on ratios, gaps and combinations.
"""
import numpy as np
import pandas as pd


def _z(s):
    return (s - s.mean()) / (s.std() + 1e-9)


# ---------------- Task 1: anti-cheat ----------------
def t1_skill_vs_experience(d):
    """Cheaters look 'too good too early': mechanical skill relative to level / account age."""
    return pd.DataFrame({
        'kd_per_level': d.kill_death_ratio / (d.level + 1),
        'hs_per_age': d.headshot_percentage / np.log1p(d.account_age_days),
        'acc_per_level': d.accuracy_score / (d.level + 1),
        'progress_per_age': d.level_progression_speed / np.log1p(d.account_age_days),
    })


def t1_mechanics_vs_tactics(d):
    """Aim bots raise mechanical stats without raising tactical ones (game sense, map knowledge)."""
    mech = pd.concat([_z(d.headshot_percentage), _z(d.accuracy_score), _z(d.kill_death_ratio),
                      -_z(d.reaction_time_ms), _z(d.crosshair_placement)], axis=1).mean(axis=1)
    tact = pd.concat([_z(d.game_sense_score), _z(d.map_knowledge), _z(d.team_play_score),
                      _z(d.buy_decision_score), _z(d.utility_usage_rate)], axis=1).mean(axis=1)
    return pd.DataFrame({'mech_score': mech, 'tact_score': tact, 'mech_minus_tact': mech - tact})


def t1_aim_consistency(d):
    """Scripts give unnaturally smooth / consistent input relative to accuracy and reaction."""
    return pd.DataFrame({
        'smooth_x_input': d.aiming_smoothness * d.input_consistency_score,
        'acc_minus_smooth': _z(d.accuracy_score) - _z(d.aiming_smoothness),
        'react_x_acc': d.reaction_time_ms * d.accuracy_score,
        'crosshair_minus_map': _z(d.crosshair_placement) - _z(d.map_knowledge),
        'spray_minus_acc': _z(d.spray_control_score) - _z(d.accuracy_score),
    })


def t1_reports_and_footprint(d):
    """Reports, device swaps and odd hours normalised by how much the player plays."""
    hours = d.avg_session_length_min * d.sessions_per_day / 60
    return pd.DataFrame({
        'reports_per_hour': d.reports_received / (hours + 0.1),
        'reports_per_friend': d.reports_received / (d.friend_network_size + 1),
        'devices_per_age': d.device_changes_count / (d.account_age_days + 1),
        'daily_hours': hours,
        'n_missing': d.isna().sum(axis=1),
    })


# ---------------- Task 2: segmentation ----------------
def t2_spend_and_time(d):
    """Whales spend per hour, grinders play a lot cheaply, socials invest in friends."""
    return pd.DataFrame({
        'spend_per_hour': d.total_spending_thb / (d.total_playtime_hours + 1),
        'monthly_x_freq': d.avg_monthly_spending * d.spending_frequency,
        'hours_per_day_active': d.total_playtime_hours / (d.play_frequency + 0.1),
        'session_x_freq': d.avg_session_duration * d.play_frequency,
        'spend_share_monthly': d.avg_monthly_spending / (d.total_spending_thb + 1),
    })


def t2_style_scores(d):
    """Social vs competitive vs collector style as averaged z-scores."""
    soc = pd.concat([_z(d.friend_count), _z(d.chat_activity_score), _z(d.friend_invites_sent),
                     _z(d.gifts_sent_received), _z(d.team_play_percentage)], axis=1).mean(axis=1)
    comp = pd.concat([_z(d.ranked_participation_rate), _z(d.tournament_entries), _z(d.win_rate_ranked)], axis=1).mean(axis=1)
    coll = pd.concat([_z(d.collection_progress), _z(d.rare_items_count), _z(d.achievement_completion_rate)], axis=1).mean(axis=1)
    pref = d[['item_type_preference_cosmetic', 'item_type_preference_performance', 'item_type_preference_social']]
    return pd.DataFrame({'social_style': soc, 'comp_style': comp, 'collector_style': coll,
                         'soc_minus_comp': soc - comp, 'pref_argmax': pref.values.argmax(axis=1)})


# ---------------- Task 3: 30-day spending ----------------
def t3_recent_trend(d):
    """Is the player spending more or less now than their lifetime average month?"""
    months = d.account_age_days / 30 + 1
    avg_month = d.historical_spending / months
    return pd.DataFrame({
        'avg_month_spend': avg_month,
        'prev_vs_avg': d.prev_month_spending / (avg_month + 1),
        'prev_minus_avg': d.prev_month_spending - avg_month,
        'prev_share_hist': d.prev_month_spending / (d.historical_spending + 1),
    })


def t3_purchase_rhythm(d):
    """Expected purchases in the next 30 days from frequency and recency."""
    gap = 30 / (d.purchase_frequency + 0.01)
    return pd.DataFrame({
        'expected_gap_days': gap,
        'overdue_ratio': d.days_since_last_purchase / (gap + 1),
        'expected_30d_value': d.avg_transaction_value * d.purchase_frequency,
        'tx_per_month': d.total_transactions / (d.account_age_days / 30 + 1),
        'hist_per_tx': d.historical_spending / (d.total_transactions + 1),
    })


def t3_engagement(d):
    """Spending tends to follow engagement: events, sessions and recency combined."""
    return pd.DataFrame({
        'weekly_hours': d.sessions_per_week * d.avg_session_length / 60,
        'event_x_prev': d.event_participation_rate * d.prev_month_spending,
        'active_recent': (d.days_since_last_login < 7).astype(float) * d.event_participation_rate,
        'login_vs_purchase_gap': d.days_since_last_purchase - d.days_since_last_login,
    })


# ---- round 2 (after round 1: only the Task 2 style/spend groups moved CV, +0.002) ----
def t2_round2_combined(d):
    """Round-1 winners together, to check that their gains add up rather than overlap."""
    return pd.concat([t2_spend_and_time(d), t2_style_scores(d)], axis=1)


def t2_round2_segment_signals(d):
    """One hand-made signal per segment: whale = paid tiers x spend, grinder = hours per baht,
    social = social share of all activity, casual = low frequency and short sessions."""
    soc = d[['friend_count', 'chat_activity_score', 'gifts_sent_received']].rank(pct=True).mean(axis=1)
    act = d[['play_frequency', 'total_playtime_hours', 'ranked_participation_rate']].rank(pct=True).mean(axis=1)
    return pd.concat([t2_spend_and_time(d), t2_style_scores(d), pd.DataFrame({
        'whale_signal': d.vip_tier.rank(pct=True) * d.avg_monthly_spending.rank(pct=True),
        'grinder_signal': d.total_playtime_hours / (d.total_spending_thb + 100),
        'social_share': soc / (soc + act + 1e-6),
        'casual_signal': -(_z(d.play_frequency) + _z(d.avg_session_duration) + _z(d.login_streak)),
    })], axis=1)
