import streamlit as st
import pandas as pd
from auth.authentication import authenticate, create_account, ensure_default_admin, logout
from database.db import init_db, dashboard_stats, list_students, list_predictions, save_prediction
from models.predict import load_model_bundle, predict_grade, get_model_metrics, feature_importance
from utils.helpers import recommendations
from utils.reports import prediction_pdf

st.set_page_config(page_title="EduPredict | Student Intelligence", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #07111f; color: #e7eef9; }
[data-testid="stHeader"] { background: rgba(7,17,31,.92); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1830 0%,#101f3b 100%); border-right:1px solid #20314d; }
[data-testid="stSidebar"] * { color:#e7eef9; }
.block-container { padding-top: 1.5rem; max-width: 1500px; }
.hero { padding: 1.5rem 1.7rem; border:1px solid #243a5a; border-radius:20px; background:linear-gradient(115deg,#101f3b 0%,#12396a 65%,#075985 100%); margin-bottom:1.2rem; }
.hero h1 { color:#fff; margin:0; font-size:2.15rem; }
.hero p { color:#c3d7f3; margin:.4rem 0 0; }
.eyebrow { color:#60a5fa; text-transform:uppercase; font-size:.76rem; letter-spacing:.14em; font-weight:800; }
.panel { background:#0e1b2e; border:1px solid #20314d; border-radius:16px; padding:1rem 1.1rem; }
.muted { color:#93a8c7; font-size:.88rem; }
div[data-testid="stMetric"] { background:#0e1b2e; border:1px solid #20314d; padding:1rem; border-radius:15px; }
div[data-testid="stMetricLabel"] { color:#9db1ce; }
div[data-testid="stMetricValue"] { color:#f3f7ff; }
div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#2563eb,#0284c7); border:0; border-radius:10px; font-weight:700; min-height:2.65rem; }
.stButton > button { border-radius:10px; }
[data-testid="stForm"] { background:#0e1b2e; border:1px solid #20314d; padding:1rem; border-radius:16px; }
input, textarea { color-scheme:dark; }
hr { border-color:#20314d; }
h2,h3,h4 { color:#f3f7ff; }
[data-testid="stDataFrame"] { border:1px solid #20314d; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

init_db()
ensure_default_admin()

def hero(title, subtitle, eyebrow="STUDENT INTELLIGENCE PLATFORM"):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def login_page():
    hero("Predict better. Support earlier.", "An academic analytics workspace for student performance, learning insights, and practical action plans.")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("### Sign in to EduPredict")
        st.caption("Use your student account or the initial administrator account.")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            user = authenticate(username.strip(), password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid username or password.")
        with st.expander("Create a student account"):
            with st.form("register_form"):
                name = st.text_input("Full name")
                email = st.text_input("Email")
                roll = st.text_input("Roll number")
                uname = st.text_input("Choose username")
                p1 = st.text_input("Password", type="password")
                p2 = st.text_input("Confirm password", type="password")
                reg = st.form_submit_button("Create student account", use_container_width=True)
            if reg:
                ok, message = create_account(name, email, roll, uname, p1, p2)
                if ok: st.success(message)
                else: st.error(message)
    with right:
        st.markdown("### Platform overview")
        st.markdown("""
        <div class="panel">
          <h4>01 · Academic prediction</h4><p class="muted">Estimate a mathematics final grade from the public UCI Student Performance dataset.</p>
          <hr><h4>02 · Student records</h4><p class="muted">Store student profiles and prediction history in a local SQLite database.</p>
          <hr><h4>03 · Admin analytics</h4><p class="muted">Review model evaluation, class-level trends, and saved records.</p>
          <hr><p class="muted">For academic demonstration only. Predictions are estimates, not decisions about a student's future.</p>
        </div>""", unsafe_allow_html=True)
        st.info("Initial admin: username `admin`, password `Admin@12345`. Change it after first login.")

def main():
    if "user" not in st.session_state:
        login_page()
        return
    user = st.session_state.user
    with st.sidebar:
        st.markdown("# 🎓 EduPredict")
        st.caption("STUDENT PERFORMANCE INTELLIGENCE")
        st.divider()
        st.write(f"**{user['full_name']}**")
        st.caption(f"{user['role'].title()} account")
        if user["role"] == "admin":
            page = st.radio("Workspace", ["Overview", "Predict performance", "Students & history", "Model evaluation", "Account"])
        else:
            page = st.radio("Workspace", ["My dashboard", "Predict performance", "My history", "Account"])
        st.divider()
        st.caption("UCI Student Performance dataset · CC BY 4.0")
        if st.button("Sign out", use_container_width=True):
            logout()
            st.rerun()

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        bundle = None
        st.error(f"Model could not be loaded: {exc}")
        st.info("Check internet access and run `python -m models.train_model` from the project folder to download/train the model.")

    if page in ("Overview", "My dashboard"):
        hero("Academic overview", "A clear snapshot of saved student records and prediction activity.", "EDUPREDICT DASHBOARD")
        stats = dashboard_stats()
        cols = st.columns(4)
        cols[0].metric("Registered students", stats["students"])
        cols[1].metric("Saved predictions", stats["predictions"])
        cols[2].metric("Average predicted grade", f"{stats['avg_grade']:.1f}/20" if stats["avg_grade"] is not None else "—")
        cols[3].metric("Students needing support", stats["support_count"])
        c1,c2 = st.columns([1.25,1])
        with c1:
            st.markdown("### Prediction activity")
            history = list_predictions(user_id=user["id"] if user["role"] != "admin" else None, limit=100)
            if history:
                df = pd.DataFrame(history)
                df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
                if "created_at" in df and df["created_at"].notna().any():
                    daily = df.dropna(subset=["created_at"]).groupby(df["created_at"].dt.date).size().reset_index(name="Predictions")
                    daily.columns = ["Date","Predictions"]
                    st.line_chart(daily.set_index("Date"))
                st.dataframe(df[["student_name","course","predicted_grade","category","created_at"]].head(8), use_container_width=True, hide_index=True)
            else:
                st.info("No predictions saved yet. Open Predict performance to create the first report.")
        with c2:
            st.markdown("### Quick actions")
            st.markdown('<div class="panel"><h4>Generate a prediction</h4><p class="muted">Enter academic indicators and get a model-based grade estimate.</p></div>', unsafe_allow_html=True)
            if st.button("Open prediction workspace", type="primary", use_container_width=True):
                st.session_state["nav_target"] = "Predict performance"
                st.rerun()
            st.markdown('<div class="panel"><h4>Use responsibly</h4><p class="muted">This dataset covers students in two Portuguese secondary schools. It may not generalize to other institutions or countries.</p></div>', unsafe_allow_html=True)
    elif page == "Predict performance":
        hero("Performance prediction", "Enter the student's current academic details to estimate the final mathematics grade.", "MODEL-ASSISTED ACADEMIC INSIGHT")
        if bundle is None:
            st.stop()
        with st.form("prediction_form"):
            c1,c2,c3 = st.columns(3)
            with c1:
                name = st.text_input("Student name", value=user["full_name"] if user["role"] != "admin" else "")
                roll = st.text_input("Roll number", value=user.get("roll_number","") if user["role"] != "admin" else "")
                studytime = st.select_slider("Weekly study time", options=[1,2,3,4], value=2, format_func=lambda x:{1:"Under 2 hours",2:"2–5 hours",3:"5–10 hours",4:"Over 10 hours"}[x])
                failures = st.number_input("Past class failures", min_value=0, max_value=4, value=0)
            with c2:
                age = st.slider("Age", 15, 22, 17)
                absences = st.number_input("School absences", min_value=0, max_value=93, value=4)
                G1 = st.slider("First Assessment Marks", 0, 20, 10)
                G2 = st.slider("Second Assessment Marks", 0, 20, 11)
            with c3:
                school_label = st.selectbox("School Type", ["Government School", "Private School"] )
                # Compatibility mapping only: the trained model still expects the original Portuguese dataset codes.
                # This display mapping is not a validated Government-vs-Private feature mapping.
                school = "GP" if school_label == "Government School" else "MS"
                sex = st.selectbox("Sex", ["F","M"])
                address = st.selectbox("Home area", ["U","R"], format_func=lambda x: "Urban" if x=="U" else "Rural")
                support = st.selectbox("Extra school support", ["no","yes"])
            st.caption("First and second assessment marks are inputs to the existing model. Important: Government/Private are display labels only; the trained model still uses Portuguese school codes internally, so its results are not validated for Indian school types.")
            submitted = st.form_submit_button("Run prediction and save report", type="primary", use_container_width=True)
        if submitted:
            if not name.strip():
                st.error("Enter a student name.")
            else:
                row = {"school":school,"sex":sex,"age":age,"address":address,"studytime":studytime,"failures":failures,"absences":absences,"G1":G1,"G2":G2,"schoolsup":support}
                pred = predict_grade(bundle, row)
                category = "Strong" if pred >= 14 else ("Developing" if pred >= 10 else "Needs support")
                tips = recommendations(row, pred)
                prediction_id = save_prediction(user_id=user["id"], student_name=name.strip(), roll_number=roll.strip(), course="Mathematics", features=row, predicted_grade=pred, category=category, recommendations=tips)
                st.session_state["last_prediction"] = {"id":prediction_id,"name":name.strip(),"roll":roll.strip(),"grade":pred,"category":category,"features":row,"tips":tips}
                st.rerun()
        result = st.session_state.get("last_prediction")
        if result:
            st.divider()
            a,b,c = st.columns(3)
            a.metric("Predicted Final Assessment Grade", f"{result['grade']:.1f} / 20")
            b.metric("Performance outlook", result["category"])
            c.metric("Report ID", f"EP-{result['id']:05d}")
            st.progress(min(1.0, max(0.0,result["grade"]/20)))
            st.markdown("### Suggested next steps")
            for tip in result["tips"]: st.write(f"• {tip}")
            pdf = prediction_pdf(result)
            st.download_button("Download PDF report", data=pdf, file_name=f"edupredict_report_{result['id']}.pdf", mime="application/pdf")
            st.download_button("Download CSV report", data=pd.DataFrame([{"student_name":result["name"],"roll_number":result["roll"],"predicted_grade":result["grade"],"category":result["category"],**result["features"]}]).to_csv(index=False).encode(), file_name=f"edupredict_report_{result['id']}.csv", mime="text/csv")
            st.caption("The estimate is not a guaranteed outcome. Review it with academic context and teacher judgement.")

            # Advanced interactive insights are calculated from the same trained model.
            st.markdown("## Advanced student insights")
            tab_graph, tab_whatif, tab_plan, tab_explain = st.tabs([
                "📊 Performance graph", "🎛️ What-if lab", "📚 Study plan", "🧠 Explain prediction"
            ])

            feature_row = dict(result["features"])
            baseline_grade = float(result["grade"])

            with tab_graph:
                st.markdown("#### Grade comparison")
                chart_df = pd.DataFrame({
                    "Assessment": ["First Assessment Marks", "Second Assessment Marks", "Predicted Final Grade"],
                    "Grade out of 20": [float(feature_row.get("G1", 0)), float(feature_row.get("G2", 0)), baseline_grade],
                }).set_index("Assessment")
                st.bar_chart(chart_df, y="Grade out of 20", use_container_width=True)
                st.caption("The predicted final grade is a model estimate. First and second assessment marks are inputs, not predictions. The dataset does not define these as six-month or half-yearly periods.")

            with tab_whatif:
                st.markdown("#### Change inputs and compare estimates")
                st.caption("These controls do not change the saved original record. They run temporary predictions with the same model.")
                w1, w2 = st.columns(2)
                with w1:
                    wi_studytime = st.select_slider(
                        "Weekly study time", options=[1, 2, 3, 4],
                        value=int(feature_row.get("studytime", 2)),
                        format_func=lambda x: {1: "Under 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "Over 10 hours"}[x],
                        key=f"whatif_study_{result['id']}"
                    )
                    wi_absences = st.slider(
                        "School absences", 0, 93, int(feature_row.get("absences", 4)),
                        key=f"whatif_absences_{result['id']}"
                    )
                with w2:
                    wi_g1 = st.slider("First Assessment Marks", 0, 20, int(feature_row.get("G1", 10)), key=f"whatif_g1_{result['id']}")
                    wi_g2 = st.slider("Second Assessment Marks", 0, 20, int(feature_row.get("G2", 11)), key=f"whatif_g2_{result['id']}")
                whatif_row = dict(feature_row)
                whatif_row.update({"studytime": wi_studytime, "absences": wi_absences, "G1": wi_g1, "G2": wi_g2})
                whatif_grade = predict_grade(bundle, whatif_row)
                delta = whatif_grade - baseline_grade
                q1, q2, q3 = st.columns(3)
                q1.metric("Original estimate", f"{baseline_grade:.1f}/20")
                q2.metric("What-if estimate", f"{whatif_grade:.1f}/20")
                q3.metric("Difference", f"{delta:+.1f} points")
                st.bar_chart(pd.DataFrame({"Scenario": ["Original", "What-if"], "Estimated grade": [baseline_grade, whatif_grade]}).set_index("Scenario"), y="Estimated grade", use_container_width=True)
                st.info("A what-if result is a model simulation, not proof that changing one factor will cause a real grade change.")

            with tab_plan:
                st.markdown("#### Suggested action plan")
                for index, tip in enumerate(result.get("tips", []), start=1):
                    st.write(f"**{index}.** {tip}")
                plan = []
                if int(feature_row.get("G2", 0)) < 10:
                    plan.append("Schedule short, regular maths revision sessions and review the topics missed in recent assessments.")
                if int(feature_row.get("studytime", 2)) <= 2:
                    plan.append("Try a realistic weekly study timetable with focused practice blocks and short breaks.")
                if int(feature_row.get("absences", 0)) >= 10:
                    plan.append("Review attendance barriers with a teacher or trusted adult and make a catch-up plan for missed lessons.")
                if int(feature_row.get("failures", 0)) > 0:
                    plan.append("Identify foundational topics that need practice and ask a teacher for targeted support.")
                if not plan:
                    plan.append("Maintain the current study routine, practise exam-style questions, and review progress with a teacher.")
                st.markdown("##### Personalised weekly actions")
                for action in plan:
                    st.checkbox(action, value=False, key=f"plan_{result['id']}_{plan.index(action)}")
                st.caption("These are rule-based study suggestions, not recommendations generated by a language model.")

            with tab_explain:
                st.markdown("#### Why did the model give this estimate?")
                st.write("This project uses a trained machine-learning regression model. It does not use a conversational AI model.")
                st.markdown("**Global feature importance**")
                try:
                    importance_df = feature_importance(bundle)
                    if not importance_df.empty and importance_df["Importance"].fillna(0).sum() > 0:
                        st.bar_chart(importance_df.set_index("Feature"), y="Importance", use_container_width=True)
                        st.caption("This describes model-level importance across the fitted model; it does not explain this individual prediction by itself.")
                    else:
                        st.info("This fitted model does not expose usable feature-importance values. Use the sensitivity comparison below instead.")
                except Exception:
                    st.info("Global feature importance is unavailable for this model. Use the sensitivity comparison below instead.")

                st.markdown("**One-input-at-a-time sensitivity check**")
                scenarios = []
                checks = [
                    ("First assessment marks + 2 points", "G1", min(20, int(feature_row.get("G1", 10)) + 2)),
                    ("Second assessment marks + 2 points", "G2", min(20, int(feature_row.get("G2", 11)) + 2)),
                    ("Study time one level higher", "studytime", min(4, int(feature_row.get("studytime", 2)) + 1)),
                    ("5 fewer absences", "absences", max(0, int(feature_row.get("absences", 4)) - 5)),
                ]
                for label, field, changed_value in checks:
                    scenario_row = dict(feature_row)
                    scenario_row[field] = changed_value
                    scenario_grade = predict_grade(bundle, scenario_row)
                    scenarios.append({"Scenario": label, "Estimated grade": scenario_grade, "Difference vs baseline": round(scenario_grade - baseline_grade, 2)})
                st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)
                st.caption("Sensitivity checks show how this model responds to hypothetical input changes. They are not causal evidence, and changes may not reflect real-world outcomes.")
    elif page in ("Students & history", "My history"):
        hero("Student records & history", "Search saved predictions and export a CSV for your academic project.", "RECORD MANAGEMENT")
        if user["role"] == "admin":
            records = list_predictions(limit=500)
        else:
            records = list_predictions(user_id=user["id"], limit=500)
        df = pd.DataFrame(records)
        if df.empty:
            st.info("No saved predictions found.")
        else:
            search = st.text_input("Search by student name or roll number")
            if search:
                df = df[df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Export history CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="edupredict_history.csv", mime="text/csv")
    elif page == "Model evaluation":
        if user["role"] != "admin":
            st.error("Admin access required.")
            st.stop()
        hero("Model evaluation", "Held-out test metrics and feature importance from the public dataset.", "MACHINE LEARNING")
        if bundle:
            metrics = get_model_metrics(bundle)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Dataset rows", metrics["rows"])
            c2.metric("MAE", f"{metrics['mae']:.2f}")
            c3.metric("RMSE", f"{metrics['rmse']:.2f}")
            c4.metric("R²", f"{metrics['r2']:.3f}")
            st.caption("Metrics are computed on a held-out test split. Dataset is small and geographically limited; results are not a guarantee of real-world accuracy.")
            imp = feature_importance(bundle)
            st.markdown("### Feature importance")
            st.bar_chart(imp.set_index("Feature"))
            st.markdown("### Dataset provenance")
            st.markdown("Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/320/student+performance). License: CC BY 4.0.")
    elif page == "Account":
        hero("Account settings", "Review your account details and update your password.", "PROFILE & SECURITY")
        st.write({"Name":user["full_name"],"Username":user["username"],"Role":user["role"],"Email":user.get("email","")})
        with st.form("password_change"):
            old = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password")
            confirm = st.text_input("Confirm new password", type="password")
            change = st.form_submit_button("Update password", type="primary")
        if change:
            from auth.authentication import change_password
            ok,msg = change_password(user["id"], old, new, confirm)
            if ok: st.success(msg)
            else: st.error(msg)

if __name__ == "__main__":
    main()
