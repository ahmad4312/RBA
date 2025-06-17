# === Streamlit Robo-Advisor Full Site with Proper Tabs ===
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.cluster.hierarchy as sch

# ============ PAGE SETTING ============
st.set_page_config(page_title="Robo-Advisor with Dr. Danial!", layout="wide")
st.title("🔮 Experience Robo-Advisor with Dr. Danial!")

# ============ DATA PREPARATION ============
stocks = [
    'Apple (AAPL)', 'Microsoft (MSFT)', 'Amazon (AMZN)', 'Tesla (TSLA)', 'Nvidia (NVDA)',
    'JPMorgan Chase (JPM)', 'Johnson & Johnson (JNJ)', 'ExxonMobil (XOM)',
    'Berkshire Hathaway (BRK.B)', 'Meta Platforms (META)'
]

np.random.seed(42)
returns = np.random.randn(252, len(stocks)) / 100

# ============ FUNCTIONS ============

def get_cluster_variance(cov, cluster):
    idx = [stocks.index(stock) for stock in cluster]
    cov_slice = cov[np.ix_(idx, idx)]
    w = np.repeat(1/len(cluster), len(cluster))
    return np.dot(w, np.dot(cov_slice, w))

def hrp_allocation(returns):
    cov = np.cov(returns, rowvar=False)
    corr = np.corrcoef(returns, rowvar=False)
    dist = np.sqrt(0.5 * (1 - corr))
    linkage = sch.linkage(dist, method='single')
    sorted_idx = sch.leaves_list(linkage)
    
    weights = pd.Series(1, index=[stocks[i] for i in sorted_idx])
    cluster_items = [[stocks[i] for i in sorted_idx]]
    
    while len(cluster_items) > 0:
        cluster_items = [i for i in cluster_items if len(i) > 1]
        if len(cluster_items) == 0:
            break
        new_cluster = []
        for cluster in cluster_items:
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            left_var = get_cluster_variance(cov, left)
            right_var = get_cluster_variance(cov, right)
            alpha = 1 - left_var / (left_var + right_var)
            weights[left] *= alpha
            weights[right] *= (1 - alpha)
            new_cluster += [left, right]
        cluster_items = new_cluster
    return weights / weights.sum()

def basic_allocation(risk):
    if risk == "Low":
        stocks_pct = [13, 13, 13, 6, 7, 11, 13, 13, 11, 0]
    elif risk == "Medium":
        stocks_pct = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    else:
        stocks_pct = [6, 6, 8, 15, 18, 8, 7, 8, 8, 6]
    stocks_pct = np.array(stocks_pct)
    stocks_pct = stocks_pct / stocks_pct.sum()
    return pd.Series(stocks_pct, index=stocks)

def smart_bionic_strategy(age, years):
    if age < 30:
        return [15, 14, 13, 10, 10, 8, 6, 6, 9, 9]
    elif age < 50:
        return [12, 12, 10, 8, 8, 10, 10, 10, 10, 10]
    else:
        return [10, 10, 8, 6, 6, 12, 14, 14, 10, 10]

def plot_portfolio_pie(weights, title):
    pastel_colors = ["#A2C4C9", "#C9DAF8", "#D9EAD3", "#F9CB9C", "#FFE599",
                     "#B6D7A8", "#CFE2F3", "#EAD1DC", "#F6B26B", "#B4A7D6"]
    fig = px.pie(
        names=weights.index,
        values=weights * 100,
        hole=0.45,
        color_discrete_sequence=pastel_colors
    )
    fig.update_traces(
        textinfo='label+percent',
        textposition='inside',
        insidetextorientation='radial',
        pull=[0.02]*len(weights)
    )
    fig.update_layout(
        height=600, width=600,
        showlegend=False,
        title_text=title,
        title_x=0.5,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=False)

def plot_growth(initial_investment, expected_return, years, key_growth):
    values = [initial_investment * (1 + expected_return) ** year for year in range(years + 1)]
    max_val = max(values)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(range(years + 1), values, marker='o')
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Projected Portfolio Growth")
    ax.text(years-3, max_val * 0.95, f"Max: ${max_val:,.0f}", ha='center', va='bottom', fontsize=10, color='green')
    st.pyplot(fig, clear_figure=True)

# ============ TAB SETUP ============
tabs = st.tabs(["About Me", "What is Robo Advisor?", "The Roots of Robo Advisor", "How it Works?", "Who Uses Robo Advisor?", "Human Advisor", "Robo Advisor", "Bionic Advisor", "Conclusion", "Group Activities"])


# === ABOUT ME TAB ===
with tabs[0]:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("ARA-0925.jpg", width=400)
    with col2:
        st.subheader("👤 About Dr. Ahmad Danial Zainudin, PhD, CFTe")
        st.markdown("""
*Finance academician with strong integration of industry experience and scholarly engagement.*

👨🏻‍💼 **Professional Experience**  
- Held various derivatives trading positions in Phillip Futures, Maybank Investment Bank, and CIMB Investment Bank.  
- Certified Financial Technician (CFTe) with experience in **investment**, **markets**, and **hedging strategies**.

🏛️ **Academic Leadership and Teaching Focus**  
- Senior Lecturer and Programme Leader for **Master in Finance** at **Asia Pacific University**.  
- Teaching fixed income and equity investment, financial management, international banking and investment, robo-advisor, and FinTech using Python.

🔬 **Research Interests**  
- Computational Finance  
- Automated Asset Allocations and Hedging Strategies  
- Econophysics and Market Microstructure

🏆 **Recognitions**  
- Top Futures Broker (Bursa Malaysia)  
- Maybank Staff Academic Award  
- IVIC Bronze Medal.
""")

    # Centered Signature
    st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: grey; margin-top: 50px;'>
        <strong>Dr. Ahmad Danial bin Zainudin </strong><br>
        School of Accounting & Finance | Asia Pacific University of Technology & Innovation (Malaysia) <br>
        danial.zainudin@apu.edu.my
    </div>
    """, unsafe_allow_html=True
    )

# === WHAT IS ROBO ADVISOR ===
with tabs[1]:
    st.markdown("<h2 style='font-size:26px; color:black;'>📚 What is a Robo-Advisor?</h2>", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("1. 📖 Introduction to Robo-Advisors"):
        st.markdown("""
<div style='font-size:16px; color:black;'>
- <b>Robo-Advisors</b> = <i>Automation + Investment Intelligence</i>.<br>
- Digital platforms offering <b>automated, algorithm-driven</b> financial planning and investing services.<br>
- Require <b>minimal human supervision</b> compared to traditional advisors.<br>
- Aim to make <b>professional-grade investment management</b> affordable and accessible to <b>everyone</b>, not just the wealthy.<br><br>

🔵 <b>In short:</b><br>
<b>Robo-Advisors = Your smart financial manager, available 24/7, without emotional bias.</b>
</div>
""", unsafe_allow_html=True)

    with st.expander("2. 🏛️ What is Digital Investment Management (DIM)?"):
        st.markdown("""
<div style='font-size:16px; color:black;'>
- <b>DIM</b> is the technological backbone powering Robo-Advisors.<br>
- It automates the entire investment journey:<br>
    - Client onboarding and profiling<br>
    - Risk appetite analysis<br>
    - Portfolio construction and asset allocation<br>
    - Continuous monitoring and automatic rebalancing<br>
    - Transparent reporting and updates<br><br>

✅ <b>DIM replaces manual wealth management with data, algorithms, and AI-driven intelligence.</b>
</div>
""", unsafe_allow_html=True)

    with st.expander("3. 🔄 How Do Robo-Advisors Work? (Simple Workflow)"):
        st.markdown("""
<div style='font-size:16px; color:black;'>
1. <b>You answer a simple questionnaire</b> (age, goals, risk comfort, investment timeline).<br>
2. <b>Algorithm constructs a diversified portfolio</b> tailored to your profile.<br>
3. <b>Ongoing monitoring and auto-rebalancing</b> to maintain optimal allocation.<br>
4. <b>Transparent, real-time reporting</b> keeps you updated on performance and risks.<br><br>

✅ <b>No guessing. No emotions. Just disciplined, data-driven investing!</b>
</div>
""", unsafe_allow_html=True)

    with st.expander("4. 🎯 Benefits of Robo-Advisors"):
        st.markdown("""
<div style='font-size:16px; color:black;'>
- <b>Low Costs:</b>  
  Robo-Advisors charge significantly lower management fees compared to traditional human advisors.<br>
- <b>Accessibility:</b>  
  Start investing with as little as <b>$100 or less</b>.<br>
- <b>Automatic Diversification:</b>  
  Investments spread across different asset classes to minimize risk.<br>
- <b>Emotion-Free Investing:</b>  
  Algorithms make decisions logically, free from panic or greed.<br>
- <b>Continuous Optimization:</b>  
  Portfolios auto-adjust to reflect changing market conditions.<br>
- <b>Transparent Fees and Reporting:</b>  
  Easy-to-understand breakdowns of costs and portfolio performance.<br><br>

✅ <b>Professional wealth management, accessible without needing to be a financial expert!</b>
</div>
""", unsafe_allow_html=True)

    with st.expander("5. ⚖️ Limitations"):
        st.markdown("""
<div style='font-size:16px; color:black;'>
- <b>Limited Personalization:</b>  
  Algorithms may struggle with highly complex or specialized financial needs.<br>
- <b>Model Risk:</b>  
  Over-reliance on historical assumptions may underperform during market shocks.<br>
- <b>Regulatory Evolution:</b>  
  Regulations for digital investment services are still evolving in many countries.<br><br>

🎯 <b>Bottom Line:</b><br>
<b>Robo-Advisors are ideal for most investors, but for complex cases, human advisors may still add value.</b>
</div>
""", unsafe_allow_html=True)

    st.success("✅ **Robo-Advisors represent the future: affordable, intelligent, and emotion-free wealth management for all investors! 🚀**")

    st.markdown("---")

    # Centered Signature
    st.markdown(
    """
    <div style='text-align: center; font-size: 13px; color: grey; margin-top: 30px;'>
        <strong>Dr. Ahmad Danial bin Zainudin</strong><br>
        School of Accounting & Finance | Asia Pacific University of Technology & Innovation (Malaysia)<br>
        danial.zainudin@apu.edu.my
    </div>
    """, unsafe_allow_html=True
    )

# === THE ROOTS OF ROBO ADVISOR ===
with tabs[2]:
    st.markdown("<h2 style='font-size:26px; color:black;'>📜 Evolution of Robo-Advisory: From Theory to Automation</h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size:20px; color:black;'>🧠 Birth of Investment Science (1952)</h3>", unsafe_allow_html=True)
    st.markdown("""
- **Harry Markowitz** introduced **Modern Portfolio Theory (MPT)**.
- Shifted investing from stock-picking to **portfolio optimization**.
- Introduced the concept of the **risk-return frontier** and **diversification principles**.
""")

    st.markdown("<h3 style='font-size:20px; color:black;'>💻 Rise of Financial Computing (1970s–1990s)</h3>", unsafe_allow_html=True)
    st.markdown("""
- Rise of **financial simulations**, **MVO (Mean-Variance Optimization)**, and **algorithmic trading**.
- Increased use of **mathematical modeling** in portfolio management.
""")

    st.markdown("<h3 style='font-size:20px; color:black;'>🌍 Trust Crisis (Post-2008)</h3>", unsafe_allow_html=True)
    st.markdown("""
- Following the **Global Financial Crisis (2008)**, investor trust declined.
- Demands for **transparency**, **low costs**, and **emotion-free investing** surged.
- Paved the way for the emergence of **Robo-Advisory platforms**.
""")

    st.markdown("<h3 style='font-size:20px; color:black;'>🚀 Maturity and Rise of Robo-Advisory (2008 Onwards)</h3>", unsafe_allow_html=True)
    st.markdown("""
- **Betterment** and **Wealthfront** emerged as pioneers of **digital wealth advisory**, making automated investing accessible to everyday investors.
- Embedded sophisticated financial models into user-friendly platforms:
  - **Modern Portfolio Theory (MPT):** Optimize returns relative to risk.
  - **Black-Litterman Model:** Blend market consensus with personalized views.
  - **Hierarchical Risk Parity (HRP):** Balance portfolio risks without relying on unstable correlations.
- The marriage of **automation** and **algorithmic intelligence** redefined wealth management.
- Democratized access to professional-grade investment strategies once reserved for the ultra-wealthy.
""")

    st.success("✅ Robo-Advisory is the natural evolution of decades of financial innovation.")

    st.markdown(
    """
    <div style='text-align: center; font-size: 13px; color: grey; margin-top: 50px;'>
        <strong>Dr. Ahmad Danial bin Zainudin</strong><br>
        School of Accounting & Finance | Asia Pacific University of Technology & Innovation (Malaysia)<br>
        danial.zainudin@apu.edu.my
    </div>
    """, unsafe_allow_html=True
    )

# === HOW IT WORKS ===
with tabs[3]:
    st.markdown("<h2 style='text-align:left; font-size:28px; color:black;'>How Does a Robo-Advisor Work?</h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size:22px; color:black;'>Step 1: Creating Your Investment Policy Statement (IPS)</h3>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:16px; color:black;'>
<ul>
<li><b>Age</b> and <b>Investment Horizon</b></li>
<li><b>Risk Tolerance</b> (Low, Medium, High)</li>
<li><b>Financial Goals</b> (e.g., Retirement, Home Purchase, Education)</li>
<li><b>Liquidity Needs</b> and <b>Time Preferences</b></li>
</ul>

<p style='font-size:14px;'>👉 The IPS acts as your personal investment blueprint — guiding portfolio construction and future rebalancing!</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<h3 style='font-size:22px; color:black;'>Step 2: Designing the Ideal Portfolio (ICM Mix)</h3>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:16px; color:black;'>
<b>Based on your IPS, Robo-Advisors build a diversified portfolio using:</b>
<ul>
<li><b>Stocks</b> — for growth potential</li>
<li><b>Bonds</b> — for stability and income</li>
<li><b>Cash</b> — for liquidity</li>
<li><b>Alternatives</b> (e.g., Real Estate, Commodities)</li>
</ul>

⚡ <b>Objective:</b> Find the right balance between maximizing returns and minimizing risks.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<h3 style='font-size:22px; color:black;'>Step 3: Intelligent Optimization Models</h3>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:16px; color:black;'>

<b>Mean-Variance Optimization (MVO)</b><br>
- Seeks the best risk-return combination by balancing expected returns and portfolio volatility.<br><br>

<b>Generalized Reduced Gradient (GRG)</b><br>
- Solves optimization problems when portfolios have multiple real-world constraints (like minimum or maximum weightings).<br><br>

<b>Hierarchical Risk Parity (HRP)</b><br>
- Clusters similar assets and allocates risks intelligently across them.<br>
- Especially powerful during volatile markets when asset correlations become unstable.

</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<h3 style='font-size:22px; color:black;'>Step 4: Continuous Monitoring & Rebalancing</h3>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:16px; color:black;'>
<ul>
<li><b>Market fluctuations</b> can cause your portfolio to drift from its target allocation.</li>
<li><b>Automatic rebalancing</b> ensures your portfolio stays aligned with your risk profile.</li>
<li><b>Emotion-free investing:</b> Avoid panic selling or greedy overexposure.</li>
</ul>

✅ Investing made smarter, simpler, and truly dynamic.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # Centered Signature
    st.markdown(
    """
    <div style='text-align: center; font-size: 13px; color: white; margin-top: 30px;'>
        <strong>Dr. Ahmad Danial bin Zainudin</strong><br>
        School of Accounting & Finance | Asia Pacific University of Technology & Innovation (Malaysia)<br>
        danial.zainudin@apu.edu.my
    </div>
    """, unsafe_allow_html=True
    )

# === WHO USES ROBO ADVISOR ===
with tabs[4]:
    st.markdown("<h2 style='font-size:26px; color:black;'>👥 Who Uses Robo-Advisors?</h2>", unsafe_allow_html=True)

    st.markdown("""
<div style='font-size:16px; color:black;'>

- 👨‍🎓 <b>Young Professionals:</b><br>
  Just starting their careers, aiming to build wealth systematically at low cost.<br><br>

- 👩‍💼 <b>Busy Executives:</b><br>
  Prefer automated, hands-off investment management while focusing on their demanding careers.<br><br>

- 👵 <b>Retirees:</b><br>
  Seeking stable income streams and conservative risk-managed portfolios.<br><br>

- 🏢 <b>Small Businesses and SMEs:</b><br>
  Managing idle cash reserves through low-cost, diversified portfolios.<br><br>

- 🌍 <b>Global Citizens:</b><br>
  Investors seeking easy cross-border access to diversified international markets.<br><br>

- 🏦 <b>Institutional Players (Pensions, Endowments, Foundations):</b><br>
  Utilizing robo-platforms for niche mandates (e.g., cash management, ESG portfolios, or tactical allocations) to complement traditional strategies.<br><br>

✅ <b>Common Theme:</b><br>
👉 Desire for simplicity, lower fees, transparency, and emotion-free investment discipline.

</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.success("✅ Robo-Advisory adoption is accelerating across individuals, businesses, and even institutional segments!")

    st.markdown(
    """
    <div style='text-align: center; font-size: 13px; color: grey; margin-top: 30px;'>
        <strong>Dr. Ahmad Danial bin Zainudin</strong><br>
        School of Accounting & Finance | Asia Pacific University of Technology & Innovation (Malaysia)<br>
        danial.zainudin@apu.edu.my
    </div>
    """, unsafe_allow_html=True
    )

# === HUMAN ADVISOR TAB ===
with tabs[5]:
    st.subheader("👤 Human Advisor (100% human decision)")

    age_human = st.number_input("Enter your age:", min_value=18, max_value=100, value=30, key="age_human")
    risk = st.selectbox("Select your risk tolerance:", ["Low", "Medium", "High"], key="risk_human")

    weights = basic_allocation(risk)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Human Advisor Portfolio")
        plot_portfolio_pie(weights, "Human Advisor Portfolio")
    with col2:
        st.subheader("📈 Portfolio Growth Projection")
        expected_return = np.dot(weights.values, np.mean(returns, axis=0)) * 252
        plot_growth(100000, expected_return, 30, key_growth="human")

    st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: grey; margin-top: 50px;'>
        <strong>Dr. Ahmad Danial bin Zainudin </strong><br>
        School of Accounting & Finance | Asia Pacific University of Technology & Innovation (Malaysia) <br>
        danial.zainudin@apu.edu.my
    </div>
    """, unsafe_allow_html=True
    )

# === ROBO ADVISOR TAB ===
with tabs[6]:
    st.subheader("🤖 Robo Advisor (Algo does the hard work)")

    age_robo = st.number_input("Enter your age:", min_value=18, max_value=100, value=30, key="age_robo")
    years_robo = st.number_input("Investment Horizon (years):", min_value=1, max_value=50, value=20, key="years_robo")
    allocation_method = st.selectbox("Select Allocation Method:", ["Basic (Human)", "Advanced (Algo)"], key="method_robo")

    if allocation_method == "Basic (Human)":
        weights = basic_allocation("Medium")
    else:
        weights = hrp_allocation(returns)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Robo Advisor Portfolio")
        plot_portfolio_pie(weights, "Robo Advisor Portfolio")
    with col2:
        st.subheader("📈 Portfolio Growth Projection")
        expected_return = np.dot(weights.values, np.mean(returns, axis=0)) * 252
        plot_growth(10000, expected_return, years_robo, key_growth="robo")

    st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: grey; margin-top: 50px;'>
        <strong>Dr. Ahmad Danial bin Zainudin </strong><br>
        School of Accounting & Finance | Asia Pacific University of Technology & Innovation (Malaysia) <br>
        danial.zainudin@apu.edu.my
    </div>
    """, unsafe_allow_html=True
    )

# === BIONIC ADVISOR TAB ===
with tabs[7]:
    st.subheader("🦾 Bionic Advisor (Human + Algo works together)")
    age_bionic = st.number_input("Enter your age:", min_value=5, max_value=100, value=30, key="age_bionic")
    years_bionic = st.number_input("Investment Horizon (years):", min_value=1, max_value=60, value=20, key="years_bionic")

    # Dynamic Strategy Selection
    if age_bionic <= 20:
        strategy = "Aggressive Growth Strategy 🚀"
        strategy_desc = "Focuses on maximum capital appreciation with high-risk, high-return assets."
    elif age_bionic <= 35:
        strategy = "Balanced Growth Strategy ⚖️"
        strategy_desc = "Balanced approach between capital growth and capital preservation."
    elif age_bionic <= 50:
        strategy = "Moderate Growth Strategy 🌿"
        strategy_desc = "Emphasizes steady growth with moderate risk exposure."
    else:
        strategy = "Conservative Growth Strategy 🛡️"
        strategy_desc = "Protects capital with cautious growth, prioritizing lower volatility assets."

    st.info(f"🔵 Based on your inputs, we suggest a **{strategy}**")

    st.markdown(f"💬 *{strategy_desc}*")

    # Portfolio generation
    bionic_weights_raw = smart_bionic_strategy(age_bionic, years_bionic)
    bionic_weights = np.array(bionic_weights_raw) / sum(bionic_weights_raw)
    weights = pd.Series(bionic_weights, index=stocks)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Bionic Advisor Portfolio")
        plot_portfolio_pie(weights, "Bionic Advisor Portfolio")
    with col2:
        st.subheader("📈 Portfolio Growth Projection")
        expected_return = np.dot(weights.values, np.mean(returns, axis=0)) * 252
        plot_growth(10000, expected_return, years_bionic, key_growth="bionic")


# === CONCLUSION TAB ===
with tabs[8]:
    st.header("🎯 Conclusion and Key Takeaways")

    st.markdown("""
---
## 📚 What Have We Learned?

- Robo-Advisors **democratize investing** — accessible for everyone.
- A well-crafted **Investment Policy Statement (IPS)** is the foundation.
- Smart technologies like **MVO**, **GRG**, **HRP** drive intelligent portfolios.
- Continuous **automatic monitoring** ensures you're always aligned.

---
## 💡 Why This Matters?

- ⏳ **Time-saving**: Less stress, more life.
- 🧠 **Emotion-free investing**: Say goodbye to fear and greed cycles.
- 🌍 **Accessible wealth management** for global citizens.

---
## 🚀 Final Thought

> "**The future of wealth management is where human dreams meet machine precision.**"

Let's work smarter, not harder! 💼🤖📈
""")

    st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: grey; margin-top: 50px;'>
        <strong>Dr. Ahmad Danial bin Zainudin </strong><br>
        School of Accounting & Finance | Asia Pacific University of Technology & Innovation (Malaysia) <br>
        danial.zainudin@apu.edu.my
    </div>
    """, unsafe_allow_html=True
    )

# === GROUP ACTIVITIES TAB ===
with tabs[9]:
    st.header("🤝 Group Activities: Build Your Own IPS")

    st.markdown("""
---
## 📋 Task: Build Your Group's Investment Policy Statement (IPS)

Work together in your group to discuss and document:

- 👥 **Group Profile:**
  - Average age of members
  - Investment experience level (Beginner, Intermediate, Advanced)

- 🎯 **Financial Goals:**
  - Example: Retirement savings, house purchase, children's education
  - Target wealth amount (e.g., \$500,000 or \$1 million)

- 📅 **Investment Horizon:**
  - Number of years to achieve your financial goals

- 💧 **Liquidity Needs:**
  - Any expected large expenses soon? (e.g., tuition fees, house deposit)

- 🔥 **Risk Tolerance:**
  - Level of risk your group can accept: Low, Moderate, or High

---
✅ Once you complete your IPS, let us test it by playing with the Human Advisor, Robo Advisor, and Bionic Advisor simulations!

---
""")

    st.success("✅ Tip: There is **no one-size-fits-all portfolio** — your IPS reflects **your group’s real goals and attitude toward risk**!")

    st.markdown(
        """
        <div style='text-align: center; font-size: 14px; color: grey; margin-top: 50px;'>
            <strong>Dr. Ahmad Danial bin Zainudin </strong><br>
            School of Accounting & Finance | Asia Pacific University of Technology & Innovation (Malaysia) <br>
            danial.zainudin@apu.edu.my
        </div>
        """, unsafe_allow_html=True
    )
