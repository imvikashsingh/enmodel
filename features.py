from feast import Entity, Feature, FeatureView, ValueType, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta

# Define entities
customer = Entity(
    name="customer",
    value_type=ValueType.INT64,
    description="Customer identifier",
)

# Define data sources
customer_source = FileSource(
    path="data/raw_customer_features.csv",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="event_timestamp",
)

transaction_source = FileSource(
    path="data/raw_transaction_features.csv",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="event_timestamp",
)

# Feature view 1: Customer demographic features
customer_demographic_fv = FeatureView(
    name="customer_demographic_features",
    entities=["customer"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="age", dtype=Int64),
        Feature(name="income", dtype=Int64),
        Feature(name="credit_score", dtype=Int64),
        Feature(name="account_age_days", dtype=Int64),
    ],
    batch_source=customer_source,
    online=True,
)

# Feature view 2: Customer banking behavior
customer_behavior_fv = FeatureView(
    name="customer_behavior_features",
    entities=["customer"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="num_products", dtype=Int64),
        Feature(name="has_credit_card", dtype=Int64),
        Feature(name="is_active_member", dtype=Int64),
        Feature(name="num_transactions_last_30d", dtype=Int64),
        Feature(name="avg_monthly_balance", dtype=Float32),
    ],
    batch_source=customer_source,
    online=True,
)

# Feature view 3: Transaction aggregated features
transaction_summary_fv = FeatureView(
    name="transaction_summary_features",
    entities=["customer"],
    ttl=timedelta(days=90),
    features=[
        Feature(name="avg_transaction_amount", dtype=Float32),
        Feature(name="total_transaction_amount", dtype=Float32),
        Feature(name="online_transaction_ratio", dtype=Float32),
        Feature(name="weekend_transaction_ratio", dtype=Float32),
    ],
    batch_source=transaction_source,
    online=True,
)

# Feature view 4: Churn target (for reference)
churn_label_fv = FeatureView(
    name="churn_label",
    entities=["customer"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="churned", dtype=Int64),
    ],
    batch_source=customer_source,
    online=True,
)
