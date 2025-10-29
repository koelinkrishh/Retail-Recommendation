# ================================================================
# app.py — Retail Recommendation Streamlit Application
# ================================================================

import os
import pickle
import json
import datetime
import pandas as pd
import numpy as np
import sqlite3
import joblib
import time
import streamlit as st

from datetime import datetime

from src.Components.data_ingestion import load_data
from src.Components.train import Retail_recommendation_model
from src.config import *
from src.logger import logging
from src.exception import CustomException



if not os.path.exists(DB_path):
    st.error(f"Database not found at path: {DB_path}")
    st.stop()

## === Utility Function ===
# @st.cache_resource
def get_connection(path=DB_path):
    """Return a sqlite3 connection."""
    try:
        conn = sqlite3.connect(path, check_same_thread=False)
        return conn
    except Exception as e:
        st.error(f"Cannot connect to database: {e}")
        return None
    
@st.cache_resource(ttl=600)
def get_table_from_db(table_name:str):
    logging.info(f"Fetching {table_name} table from database")
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def _to_json(obj):
    logging.info("Converting object to json")
    if isinstance(obj, (list, dict)):
        return json.dumps(obj)
    return obj

# === Load model ===
@st.cache_resource
def load_model():
    logging.info("Loading best model")
    # Create fresh model object
    model = Retail_recommendation_model(
        Half_life=7,
        Time_period=100,   # Use your logic if needed
        filter_items=False,
        include_description=True,
        iteration_bar=False,
    )
    
    # Load weights
    weights = load_data(os.path.join(Exp_dir, "Final_model_weights.pkl"), form='pickle')
    model.weights = weights.copy()
    model.d_effect = weights.get('d_effect', 0.0)
    
    return model

# ---------------- Helper: Add to Cart Callback ----------------
def add_to_cart_callback(item):
    """Add an item to basket and refresh."""
    try:
        if "Current_basket" not in st.session_state:
            st.session_state.Current_basket = pd.DataFrame(columns=["StockCode", "Description", "Quantity", "Price", "Total_amount"])
        
        # Validate item
        if not isinstance(item, dict) or "StockCode" not in item:
            logging.warning("Invalid item format.")
            st.warning("cannot add this item, Invalid format")
            return
        
        qty = int(item.get("Quantity", 1))
        price = float(item.get("Price", 0.0))
        
        
        new_row = pd.DataFrame([{
        "StockCode": str(item["StockCode"]),
        "Description": item.get("Description", ""),
        "Quantity": qty,
        "Price": price,
        "Total_amount": qty * price,
        }])

        st.session_state.Current_basket = pd.concat(
            [st.session_state.Current_basket, new_row],
            ignore_index=True
        )
        # Log and confirm
        logging.info(f"Item added to cart: {item}")
        st.session_state.last_added_item = item  # track last addition
        st.toast(f"🛒 Added '{item['Description']}' to cart!", icon="✅")

        # Trigger cart refresh (Streamlit reruns automatically after state change)
        st.session_state.cart_updated = True

    except Exception as e:
        logging.error(f"Error adding item: {e}")
        st.warning("Could not add item to cart.")


Rec = load_model()

# Loading all datasets needed
transactions = get_table_from_db("Transactions")
customer_data = get_table_from_db("Customers")
items_data = get_table_from_db("Items")
baskets_data = get_table_from_db("Baskets")
all_customer = customer_data['Customer_ID'].unique()

basket_cols = ["StockCode", "Description", "Quantity", "Price", "Total_amount"]

# ================================================================
# Streamlit App Configuration
# ================================================================
st.set_page_config(page_title="Retail Recommendation App", page_icon=":shopping_cart:", layout="wide")
st.title("Retail Recommendation System")

# -------------------------------------------------------------------
## APP DESIGNING:
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* === GLOBAL APP STYLE (Stormy Morning Theme) === */
    .stApp {
        /* background: linear-gradient(160deg, #1B2735 0%, #2C3E50 100%); # Nordic blue */
        background: linear-gradient(160deg, #0D1F1E 0%, #203A43 50%, #2C5364 100%); /* emerald mist */

        color: #E8E8E8;
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }

    h1, h2, h3 {
        color: #CFCFEA;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* === Buttons === */
    div.stButton > button {
        background-color: #9B7EDE;
        color: #FFFFFF;
        border-radius: 10px;
        border: none;
        padding: 8px 18px;
        font-weight: 600;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 2px 8px rgba(155, 126, 222, 0.3);
    }
    div.stButton > button:hover {
        background-color: #866AD9;
        transform: scale(1.05);
    }

    /* === Number Inputs === */
    input[type=number] {
        background-color: #2E2F36;
        border: 1px solid #4B4C57;
        border-radius: 6px;
        color: #E8E8E8;
        padding: 6px 8px;
    }

    /* === Recommendation Cards === */
    .recommend-card {
        background-color: rgba(42, 43, 50, 0.85);
        border: 1px solid #4B4C57;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out;
    }
    .recommend-card:hover {
        border-color: #9B7EDE;
        box-shadow: 0 4px 10px rgba(155, 126, 222, 0.25);
    }
    .recommend-card strong {
        color: #CFCFEA;
        font-size: 15px;
    }
    .recommend-card span.price {
        color: #D4AF37;
        font-weight: bold;
    }

    /* === DataFrame Styling === */
    .stDataFrame {
        background-color: rgba(46, 47, 54, 0.85);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        color: #E8E8E8;
    }

    /* === Alerts === */
    .stAlert {
        background-color: rgba(73, 71, 90, 0.5) !important;
        border-left: 5px solid #9B7EDE;
        color: #E8E8E8;
        border-radius: 8px;
    }

    /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background-color: #2A2B32;
        color: #E8E8E8;
        border-right: 1px solid #3F3F4A;
    }
    section[data-testid="stSidebar"] h1, h2, h3, p, div {
        color: #E8E8E8;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ------------------------
# Session state initialization
# ------------------------
if "page" not in st.session_state:
    logging.info("Enter Home login Page.")
    st.session_state.page = "login"
# Ensure basket stored consistently
if "Current_basket" not in st.session_state:
    st.session_state.Current_basket = pd.DataFrame(columns=basket_cols)
if "invoice" not in st.session_state:
    try:
        last_invoice = int(baskets_data["Invoice"].max()) if (baskets_data is not None and not baskets_data.empty) else None
    except Exception:
        last_invoice = 10000
    st.session_state.invoice = last_invoice + 1
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None

# ==========================================================
# LOGIN PAGE
# ==========================================================
if st.session_state.page == 'login':
    st.write(" Welcome! Plearse choose how you want to continue: ")
    
    with st.form("Customer Login:"):
        st.header("Customer Login")
        # choose b/w login or anonymous purchase
        customer_choice = st.radio("Login or New Customer", ("Login", "Anonymous", "Create new ID"))
    
        customer_id = None
    
        if customer_choice == "Create new ID":
            new_id = int(customer_data["Customer_ID"].max()) + 1
            customer_id = new_id        
        elif customer_choice == "Login":
            entered_id = st.text_input("Enter your Customer ID:")
            if entered_id.strip() != "":
                try:
                    customer_id = int(entered_id)
                except ValueError:
                    st.warning("Please enter a valid numeric Customer ID.")
        else: # Anonymous
            st.info("You are continuing as an anonymous user.")
            customer_id = -1 # Use a special ID for anonymous users
            
        submit = st.form_submit_button("Continue")
    
    if submit:
        if customer_id is not None:
            st.session_state.customer_id = customer_id
            st.session_state.page = 'main'
        else:
            st.warning("Please enter a valid Customer ID.")

## ===== MAIN PAGE =====
elif st.session_state.page == 'main':
    logging.info("Enter Main Page")
    customer_id = st.session_state.customer_id
    
    if customer_id == -1:
        st.sidebar.info("You are continuing as an anonymous user.")
    else:
        st.sidebar.success(f"Logged in as Customer ID: {customer_id}")
        
    # --------- Cart Management -------------
    st.subheader(f"Current Basket -> ID: #{st.session_state.invoice} , customer ID: {customer_id}")
    basket = st.session_state.Current_basket.copy()
    
    if not basket.empty: # Items already selected
        basket['Total_amount'] = basket['Quantity'] * basket['Price']
        total_qty = basket['Quantity'].sum()
        total_amount = basket['Total_amount'].sum()

        st.dataframe(basket, width='content') # Use same width as parent container
        st.write(f"**Total Quantity:** {total_qty} **Total Amount:** {round(total_amount,6)}")
    else:
        st.info("Basket is empty.")
    
    st.divider() # Add divider in page

    # === Search or browse items ===
    logging.info("Searching for items")
    st.subheader("Search for products")
    
    # --- Ensure basket initialized once ---
    if "Current_basket" not in st.session_state:
        st.session_state.Current_basket = pd.DataFrame(columns=["StockCode", "Description", "Quantity", "Price", "Total_amount"])
    
    # --- Session state setup ---
    if "search_term" not in st.session_state:
        st.session_state.search_term = ""
    if "selected_item_code" not in st.session_state:
        st.session_state.selected_item_code = None
    
    # We use a small number of results shown; input doesn't autosubmit to avoid reruns
    with st.form("search_form", clear_on_submit=False):
        search_term = st.text_input("Enter StockCode or words from Description:", value=st.session_state.search_term)
        search_submitted = st.form_submit_button("Search")
    
    if search_submitted: # persist search term on submittion
        st.session_state.search_term = search_term.strip()
        
    matching_items = pd.DataFrame()
    if st.session_state.search_term:
        # time.sleep(0.1) # small rebounce delay
        words = st.session_state.search_term.lower().split()
        
        # 1. check exact StockCode match
        exact_match = items_data[items_data["StockCode"].astype(str) == st.session_state.search_term]
        if not exact_match.empty:
            matching_items = exact_match.copy()
        else: # 2. Match words from descriptions containing  all/any words
            def desc_match(desc, Type='all'):
                desc = str(desc).lower()
                if Type=='all':
                    match = all(w in desc for w in words)
                elif Type=='any':
                    match = any(w in desc for w in words)
                return match
            matching_items = items_data[items_data["Description"].apply(desc_match)]

        # Display dropdown suggestion if multiple matches
        if not matching_items.empty:
            matching_items = matching_items.copy()
            matching_items["Description"] = matching_items["Description"].str.replace("|", " ")
            
            # show limited results
            displayy = matching_items.sort_values("Frequency", ascending=False).head(20).reset_index(drop=True)
            selected_option = st.selectbox(
                "Select an item:",
                options=[f"{row.StockCode} := {row.Description}" for _, row in displayy.iterrows()],
            )

            if selected_option:
                selected_code = selected_option.split(" :=")[0]
                st.session_state.selected_item_code = selected_code
                
                selected_row = displayy[displayy["StockCode"] == selected_code].iloc[0]
                st.markdown(f"**Selected:** {selected_row['StockCode']} — {selected_row['Description']}")
                
                qty = st.number_input("Quantity", min_value=1, value=1, step=1)
                
                st.button(
                    "Add to Cart",
                    key=f"add_{selected_row['StockCode']}",
                    on_click=add_to_cart_callback,
                    args=({
                        "StockCode": selected_row["StockCode"],
                        "Description": selected_row["Description"],
                        "Price": float(selected_row.get("Current_Price", selected_row.get("Price", 0.0))),
                        "Quantity": qty,
                    },)
                )

        else:
            st.info("No matching items found.")

    st.divider() # another divide between sections
        
    # === Recommendations ===
    logging.info("Getting personalized Recommendations")
    st.subheader(" Recommendations ")
    
    try:
        # Convert basket to model input (your existing correct logic)# Convert basket to model input (robust & compatible format)
        if not st.session_state.Current_basket.empty:
            def prepare_basket_for_model():
                """Convert current basket to model-compatible format."""
                try:
                    basket_df = st.session_state.Current_basket
                    if basket_df.empty:
                        return pd.DataFrame([{"Customer_ID": st.session_state.customer_id, "StockCode": []}])

                    stock_list = basket_df["StockCode"].tolist()
                    return pd.DataFrame({
                        "Customer_ID": [st.session_state.customer_id],
                        "StockCode": [stock_list]  # list inside a list → valid for one basket
                    })
                except Exception as e:
                    logging.error(f"Error preparing basket for model: {str(e)}")
                    return pd.DataFrame([{"Customer_ID": st.session_state.customer_id, "StockCode": []}])
            
            try:
                basket_ready = prepare_basket_for_model()

                # Important: ensure StockCode column dtype is object and contains lists
                if 'StockCode' in basket_ready.columns:
                    basket_ready['StockCode'] = basket_ready['StockCode'].apply(lambda x: list(x) if not isinstance(x, list) else x)


                rec_df = Rec.recommend(baskets=basket_ready, top_n=20)
                # rec_df is now a DataFrame of recommendations for the single basket
            except Exception as e:
                st.error(f"Model recommendation unavailable -> basket format: {str(e)}")
                logging.error(f"Recommendation error in app.py: {str(e)}")
        else:
            rec_df = None

        if rec_df is not None and not rec_df.empty:
            rec_df["Description"] = rec_df["Description"].str.replace("|", " ", regex=False)
            st.markdown("#### Top 20 Personalized Recommendations")

            for _, row in rec_df.iterrows():
                col1, col2, col3 = st.columns([6, 2, 2])

                with col1:
                    st.markdown(
                        f"""
                        <div class="recommend-card">
                            <strong>{row['Description']} - </strong>
                            <span class="price">{row['Current_Price']:.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with col2:
                    qty = st.number_input(
                        f"Quantity",
                        min_value=1,
                        max_value=20,
                        value=1,
                        key=f"qty_rec_{row['StockCode']}"
                    )

                with col3:
                    st.button(
                        "Add",
                        key=f"add_rec_{row['StockCode']}",
                        on_click=add_to_cart_callback,
                        args=({
                            "StockCode": row["StockCode"],
                            "Description": row["Description"],
                            "Price": float(row["Current_Price"]),
                            "Quantity": qty
                        },)
                    )
        else:
            st.info("No recommendations yet — try adding items to your basket.")

    except Exception as e:
        logging.error(f"Recommendation error: {e}")
        st.warning(f"Model recommendation unavailable: {e}")

    st.divider()
    
    ## Top 5 most frequent items
    logging.info("Displaying top frequent items")
    st.subheader("Top 5 Frequent Items")
    top5_freq = items_data.sort_values("Frequency", ascending=False).head(5)
    top5_freq.loc[:,"Description"] = top5_freq["Description"].str.replace("|", " ", regex=False)

    for _, row in top5_freq.iterrows():
        col1, col2, col3 = st.columns([6, 2, 2])

        with col1:
            st.markdown(
                f"""
                <div class="recommend-card">
                    <strong>{row['Description']} - </strong>
                    <span class="price">{row['Current_Price']:.2f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:
            qty = st.number_input(
                f"Quantity",
                min_value=1,
                max_value=20,
                value=1,
                key=f"qty_freq_{row['StockCode']}"
            )

        with col3:
            new_row = {
                "StockCode": row["StockCode"],
                "Description": row["Description"],
                "Price": float(row.get("Current_Price", 0.0)),
                "Quantity": qty
            }
            st.button(
                "Add",
                key=f"add_freq_{row['StockCode']}",
                on_click=lambda item=new_row: add_to_cart_callback(item),
            )
    
    st.divider()
    
    # === Checkout ===
    logging.info("Finished purchasing. Initializing Checkout")
    st.header("Checkout")
    
    if st.button("Checkout"):
        if not st.session_state.Current_basket.empty:
            # Ensure numeric columns and recalc totals
            st.session_state.Current_basket['Total_amount'] = st.session_state.Current_basket['Quantity'] * st.session_state.Current_basket['Price']
            total = float(st.session_state.Current_basket['Total_amount'].sum())
            total_qty = int(st.session_state.Current_basket['Quantity'].sum())
            st.success(f"Basket checked out. Total bill: {total:.2f}")
            
             # --- Save current basket for display on next page ---
            st.session_state.checkout_summary = {
                "basket_df": st.session_state.Current_basket.copy(),
                "total": total,
                "total_qty": total_qty,
                "invoice": st.session_state.invoice,
                "customer_id": st.session_state.customer_id,
            }

            
            # Display current time but work off current_date
            purchase_date = datetime.now().strftime("%Y-%m-%d")
            purchase_time = datetime.now().strftime("%H:%M:%S")
            
            invoice = st.session_state.invoice
            customer = customer_id
            # --- Compute aggregated transaction data ---
            basket_df = st.session_state.Current_basket.copy()
            
            # Lists for grouped columns
            stock_codes = basket_df["StockCode"].astype(str).tolist()
            descriptions = basket_df["Description"].tolist()
            quantities = basket_df["Quantity"].tolist()
            prices = basket_df["Price"].tolist()
            total_amount = total
            num_products = len(stock_codes)


            # -- Saving This Transaction to Database --
            # conn = get_connection()
            # cursor = conn.cursor()
            # cursor.execute("""
            #     INSERT INTO Baskets (
            #         Invoice, Customer_ID, Purchase_Time, Purchase_Date,
            #         StockCode, Description, Quantity, Price, Total_amount, Num_products
            #     )
            #     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            # """, (
            #     invoice,
            #     customer,
            #     purchase_time,
            #     purchase_date,
            #     json.dumps(stock_codes),      # store as stringified list
            #     json.dumps(descriptions),     # store as stringified list
            #     json.dumps(quantities),       # store as stringified list
            #     json.dumps(prices),           # store as stringified list
            #     total_amount,
            #     num_products
            # ))
            
            # conn.commit()
            # conn.close()

            st.success("Transaction completed and saved to database!")
            st.success(f"✅ Basket checked out. Total bill: {total_amount:.2f}")

            # Reset session for new transaction
            st.session_state.invoice += 1
            
            st.session_state.page = "checkout_summary"
            
            time.sleep(2) # Wait for 3 seconds
            st.rerun() # Rerun the app to start a new transaction
    else:
        st.warning("Basket is empty — add items first!")
# ==========================================================
# CHECKOUT SUMMARY PAGE
# ==========================================================
elif st.session_state.page == "checkout_summary":
    logging.info("Displaying checkout summary page")
    summary = st.session_state.get("checkout_summary", None)

    if summary is None:
        st.warning("No checkout data available. Please add items and checkout again.")
    else:
        st.header(f"Invoice Summary — ID #{summary['invoice']}")
        st.subheader(f"Customer ID: {summary['customer_id']}")
        st.dataframe(summary["basket_df"], use_container_width=True)

        st.markdown(f"### 🧾 Total Quantity: **{summary['total_qty']}**")
        st.markdown(f"### 💰 Total Bill: **{summary['total']:.2f}**")

        st.success("Thank you for your purchase! Please proceed to payment counter.")

        # Reset basket AFTER showing summary
        if st.button("Return to Home"):
            st.session_state.Current_basket = pd.DataFrame(columns=["StockCode", "Description", "Quantity", "Price", "Total_amount"])
            st.session_state.page = "main"
            st.rerun()

