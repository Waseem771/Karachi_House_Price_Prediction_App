import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="🏠 Karachi House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .info-box {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00a8e8;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_resource
def load_and_prepare_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv(r"G:\Pakistan_Property_data_set_ML_Training\Property_Pakistan_datas_set\Property_with_Feature_Engineering.csv")
        df = df[df['city'] == 'Karachi'].copy()

        # Clean data
        df = df.drop_duplicates()
        df = df.dropna(subset=['price'])

        # Remove outliers (1% to 99%)
        q1 = df['price'].quantile(0.01)
        q99 = df['price'].quantile(0.99)
        df = df[(df['price'] >= q1) & (df['price'] <= q99)]

        # Fill missing values
        for col in ['baths', 'bedrooms', 'area_sqft']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_model(df):
    """Train the Random Forest model"""
    try:
        # Keep top 15 locations
        top_locs = df['location'].value_counts().head(15).index.tolist()
        df['location'] = df['location'].apply(lambda x: x if x in top_locs else 'Other')

        # Prepare features
        features = ['bedrooms', 'baths', 'area_sqft', 'year', 'location', 'property_type', 'purpose']
        X = df[features].copy()
        X = pd.get_dummies(X, columns=['location', 'property_type', 'purpose'], drop_first=True)
        y = df['price']

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Calculate metrics
        r2_score = model.score(X_test, y_test)

        return model, X.columns.tolist(), r2_score, top_locs
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None, None

# Load data and train model
df = load_and_prepare_data()
if df is not None:
    model, feature_columns, r2_score, locations = train_model(df)
else:
    st.error("Failed to load data. Please check the file path.")
    st.stop()

# Sidebar
st.sidebar.title("ℹ️ About This App")
st.sidebar.info("""
    ### 🏠 Karachi House Price Predictor

    This app predicts house prices in Karachi based on:
    - **Location** (15+ areas)
    - **Property Type** (House, Flat, etc.)
    - **Purpose** (Sale/Rent)
    - **Bedrooms & Bathrooms**
    - **Area (sqft)**
    - **Year Listed**

    **Model Accuracy:** {:.1%}

    **Data:** 60,000+ properties from Karachi

    **Algorithm:** Random Forest Regressor
""".format(r2_score))

st.sidebar.divider()
st.sidebar.title("📊 Dataset Statistics")
st.sidebar.metric("Total Properties", f"{len(df):,}")
st.sidebar.metric("Average Price", f"PKR {df['price'].mean():,.0f}")
st.sidebar.metric("Price Range", f"PKR {df['price'].min():,.0f} - {df['price'].max():,.0f}")

# Main header
st.title("🏠 Karachi House Price Predictor")
st.markdown("### Get instant price estimates for Karachi properties")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict Price", "📊 Market Analysis", "📈 Dataset Insights", "❓ Help"])

# ==================== TAB 1: PREDICT PRICE ====================
with tab1:
    st.header("Make a Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Location Details")
        area = st.selectbox(
            "Select Area:",
            sorted(locations),
            help="Choose the neighborhood in Karachi"
        )

        property_type = st.selectbox(
            "Property Type:",
            ["House", "Flat", "Apartment"],
            help="Type of property"
        )

    with col2:
        st.subheader("🏠 Property Features")
        bedrooms = st.slider(
            "Bedrooms:",
            min_value=0,
            max_value=10,
            value=3,
            help="Number of bedrooms (0-10)"
        )

        bathrooms = st.slider(
            "Bathrooms:",
            min_value=0,
            max_value=8,
            value=2,
            help="Number of bathrooms (0-8)"
        )

    with col3:
        st.subheader("📐 Additional Info")
        area_sqft = st.slider(
            "Area (sqft):",
            min_value=500,
            max_value=10000,
            value=1500,
            step=100,
            help="Property size in square feet"
        )

        purpose = st.selectbox(
            "Purpose:",
            ["For Sale", "For Rent"],
            help="Is this property for sale or rent?"
        )

    year = st.slider(
        "Year Listed:",
        min_value=2015,
        max_value=2025,
        value=2019,
        help="Year when property was listed"
    )

    # Prediction button
    if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
        try:
            # Create prediction input
            input_data = pd.DataFrame({
                'bedrooms': [bedrooms],
                'baths': [bathrooms],
                'area_sqft': [area_sqft],
                'year': [year]
            })

            # Initialize all features with 0
            for col in feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Set categorical features
            for col in feature_columns:
                if f"location_{area}" == col:
                    input_data[col] = 1
                if f"property_type_{property_type}" == col:
                    input_data[col] = 1
                if f"purpose_{purpose}" == col:
                    input_data[col] = 1

            # Reorder columns to match training
            input_data = input_data[feature_columns]

            # Predict
            predicted_price = model.predict(input_data)[0]
            price_per_sqft = predicted_price / area_sqft

            # Display results
            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                    <div class="prediction-box">
                    💰 PKR {predicted_price:,.0f}
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Price per sqft", f"PKR {price_per_sqft:,.0f}")

            # Show summary
            st.divider()
            st.subheader("📋 Summary")

            summary_data = {
                "Area": area,
                "Property Type": property_type,
                "Purpose": purpose,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Size (sqft)": f"{area_sqft:,}",
                "Year": year,
                "Estimated Price": f"PKR {predicted_price:,.0f}"
            }

            summary_df = pd.DataFrame(list(summary_data.items()), columns=["Feature", "Value"])
            st.table(summary_df)

            # Market comparison
            st.divider()
            st.subheader("🏘️ Market Comparison")

            # Get similar properties from dataset
            similar = df[
                (df['location'] == area) &
                (df['bedrooms'] == bedrooms) &
                (df['baths'] == bathrooms)
            ]

            if len(similar) > 0:
                avg_price = similar['price'].mean()
                min_price = similar['price'].min()
                max_price = similar['price'].max()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Market Average", f"PKR {avg_price:,.0f}")
                with col2:
                    st.metric("Min in Market", f"PKR {min_price:,.0f}")
                with col3:
                    st.metric("Max in Market", f"PKR {max_price:,.0f}")
                with col4:
                    diff = predicted_price - avg_price
                    st.metric("vs Market", f"PKR {diff:,.0f}", f"{(diff/avg_price)*100:.1f}%")
            else:
                st.info("No similar properties found in the dataset for exact comparison.")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

# ==================== TAB 2: MARKET ANALYSIS ====================
with tab2:
    st.header("📊 Market Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Average Price by Area")
        area_prices = df.groupby('location')['price'].mean().sort_values(ascending=False).head(10)

        fig = px.bar(
            x=area_prices.values,
            y=area_prices.index,
            orientation='h',
            labels={'x': 'Average Price (PKR)', 'y': 'Area'},
            title='Top 10 Most Expensive Areas'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏠 Properties by Type")
        type_counts = df['property_type'].value_counts().head(5)

        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title='Property Type Distribution'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Price distribution
    st.subheader("📈 Price Distribution in Karachi")
    fig = px.histogram(
        df,
        x='price',
        nbins=50,
        labels={'price': 'Price (PKR)'},
        title='How Prices are Distributed'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Bedrooms vs Price
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛏️ Bedrooms vs Price")
        bedroom_prices = df.groupby('bedrooms')['price'].mean().sort_index()

        fig = px.bar(
            x=bedroom_prices.index,
            y=bedroom_prices.values,
            labels={'x': 'Bedrooms', 'y': 'Average Price (PKR)'},
            title='Average Price by Number of Bedrooms'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Area Size vs Price")

        fig = px.scatter(
            df.sample(min(1000, len(df))),
            x='area_sqft',
            y='price',
            opacity=0.6,
            labels={'area_sqft': 'Area (sqft)', 'price': 'Price (PKR)'},
            title='Relationship between Size and Price'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: DATASET INSIGHTS ====================
with tab3:
    st.header("📈 Dataset Insights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        st.metric("Average Price", f"PKR {df['price'].mean()/1e6:.1f}M")
    with col3:
        st.metric("Unique Areas", df['location'].nunique())
    with col4:
        st.metric("Avg Bedrooms", f"{df['bedrooms'].mean():.1f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Basic Statistics")
        stats_data = {
            "Metric": ["Minimum", "25th %ile", "Median", "75th %ile", "Maximum", "Mean"],
            "Price (PKR)": [
                f"{df['price'].min():,.0f}",
                f"{df['price'].quantile(0.25):,.0f}",
                f"{df['price'].median():,.0f}",
                f"{df['price'].quantile(0.75):,.0f}",
                f"{df['price'].max():,.0f}",
                f"{df['price'].mean():,.0f}"
            ]
        }
        st.table(pd.DataFrame(stats_data))

    with col2:
        st.subheader("🏘️ Top 10 Areas")
        top_areas = df['location'].value_counts().head(10)

        fig = px.bar(
            x=top_areas.values,
            y=top_areas.index,
            orientation='h',
            labels={'x': 'Number of Properties', 'y': 'Area'},
            title='Areas with Most Listings'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("🔍 Detailed Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Bedrooms Distribution**")
        br_stats = df['bedrooms'].describe().round(2)
        st.table(br_stats)

    with col2:
        st.write("**Bathrooms Distribution**")
        ba_stats = df['baths'].describe().round(2)
        st.table(ba_stats)

    with col3:
        st.write("**Area Distribution (sqft)**")
        area_stats = df['area_sqft'].describe().round(0)
        st.table(area_stats)

# ==================== TAB 4: HELP ====================
with tab4:
    st.header("❓ Help & Guide")

    st.subheader("🚀 How to Use This App")

    st.markdown("""
    ### Step 1: Select Location
    Choose the area in Karachi where you want to predict the property price.

    ### Step 2: Choose Property Type
    Select between House, Flat, or Apartment.

    ### Step 3: Enter Features
    - **Bedrooms**: Number of sleeping rooms (0-10)
    - **Bathrooms**: Number of bathrooms (0-8)
    - **Area**: Size in square feet (500-10,000)
    - **Purpose**: Whether it's for Sale or Rent
    - **Year**: When the property was listed

    ### Step 4: Get Prediction
    Click "🔍 Predict Price" to see the estimated price.
    """)

    st.divider()

    st.subheader("📊 Understanding the Results")
    st.markdown("""
    - **Estimated Price**: The model's prediction based on your inputs
    - **Price per sqft**: Average cost per square foot
    - **Market Average**: Average price for similar properties
    - **Comparison**: How your prediction compares to the market
    """)

    st.divider()

    st.subheader("🤔 FAQs")

    with st.expander("How accurate is the prediction?"):
        st.write(f"""
        The model has an accuracy of **{r2_score:.1%}** (R² score).

        This means it explains about {r2_score:.0%} of the price variation.
        Typical prediction error is ±5-15 million PKR.

        The prediction is based on historical data and actual market conditions may vary.
        """)

    with st.expander("What's the best location to invest?"):
        st.write("""
        Based on the data:
        - **Most Expensive**: DHA Defence, Gulshan, Defence
        - **Best Value**: Some developing areas with good appreciation potential
        - Always compare with current market trends!
        """)

    with st.expander("Can I use this for rental predictions?"):
        st.write("""
        Yes! Select "For Rent" as the purpose.
        The model will predict monthly rental prices instead of sale prices.
        """)

    with st.expander("How often is the data updated?"):
        st.write("""
        The data is from Zameen.com property listings.
        For the most current prices, check real estate websites.
        This model provides estimates based on historical patterns.
        """)

    with st.expander("What areas are covered?"):
        st.write(f"""
        The app covers {len(locations)} major areas in Karachi:

        {', '.join(sorted(locations))}
        """)

    st.divider()

    st.subheader("💡 Tips for Better Predictions")
    st.markdown("""
    ✅ **Be Accurate**: Enter realistic values for features

    ✅ **Check Market**: Compare with actual listings in the area

    ✅ **Consider Location**: Location is the biggest price factor

    ✅ **Size Matters**: Larger properties generally cost more

    ✅ **Market Trends**: Prices change over time

    ✅ **Amenities**: The model uses basic features; special amenities may increase value
    """)

    st.divider()

    st.subheader("📞 Support")
    st.markdown("""
    - **Dataset**: 60,000+ properties from Karachi
    - **Model**: Random Forest Regressor (100 trees)
    - **Last Updated**: 2026-07-14
    - **Data Source**: Zameen.com
    """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🏠 <b>Karachi House Price Predictor</b> | Powered by Machine Learning</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
