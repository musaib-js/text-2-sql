-- --------------------------
-- 1. CASA Accounts Information
-- --------------------------
CREATE TABLE casa_accounts (
    account_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    account_number VARCHAR(20) UNIQUE NOT NULL,
    account_type VARCHAR(20),
    branch_id INT,
    balance NUMERIC,
    currency VARCHAR(10),
    opened_date DATE,
    status VARCHAR(20),
    last_activity_date DATE,
    interest_rate NUMERIC,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- --------------------------
-- 2. Debit Cards Information
-- --------------------------
CREATE TABLE debit_cards (
    card_id SERIAL PRIMARY KEY,
    account_id INT NOT NULL,
    card_number VARCHAR(16) UNIQUE NOT NULL,
    card_type VARCHAR(20),
    issued_date DATE,
    expiry_date DATE,
    cvv VARCHAR(4),
    status VARCHAR(20),
    daily_limit NUMERIC,
    monthly_limit NUMERIC,
    branch_id INT,
    FOREIGN KEY (account_id) REFERENCES casa_accounts(account_id)
);

-- --------------------------
-- 3. Day-wise EOP (End-of-Period) Balance
-- --------------------------
CREATE TABLE eop_balances (
    balance_id SERIAL PRIMARY KEY,
    account_id INT NOT NULL,
    date DATE NOT NULL,
    opening_balance NUMERIC,
    closing_balance NUMERIC,
    credits NUMERIC,
    debits NUMERIC,
    avg_daily_balance NUMERIC,
    currency VARCHAR(10),
    branch_id INT,
    FOREIGN KEY (account_id) REFERENCES casa_accounts(account_id)
);

-- --------------------------
-- 4. Insurance Policy Accounts
-- --------------------------
CREATE TABLE insurance_policies (
    policy_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    policy_number VARCHAR(30) UNIQUE NOT NULL,
    policy_type VARCHAR(50),
    coverage_amount NUMERIC,
    premium_amount NUMERIC,
    start_date DATE,
    end_date DATE,
    status VARCHAR(20),
    nominee_name VARCHAR(100),
    agent_id INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- --------------------------
-- 5. Loan Accounts
-- --------------------------
CREATE TABLE loan_accounts (
    loan_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    loan_account_number VARCHAR(20) UNIQUE NOT NULL,
    loan_type VARCHAR(50),
    principal_amount NUMERIC,
    outstanding_amount NUMERIC,
    interest_rate NUMERIC,
    tenure_months INT,
    start_date DATE,
    end_date DATE,
    status VARCHAR(20),
    branch_id INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- --------------------------
-- 6. Loan Accounts Historical
-- --------------------------
CREATE TABLE loan_accounts_history (
    history_id SERIAL PRIMARY KEY,
    loan_id INT NOT NULL,
    date DATE NOT NULL,
    principal_outstanding NUMERIC,
    interest_outstanding NUMERIC,
    total_due NUMERIC,
    payment_made NUMERIC,
    overdue_amount NUMERIC,
    status VARCHAR(20),
    remarks TEXT,
    FOREIGN KEY (loan_id) REFERENCES loan_accounts(loan_id)
);

-- --------------------------
-- 7. Credit Cards Accounts
-- --------------------------
CREATE TABLE credit_cards (
    card_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    card_number VARCHAR(16) UNIQUE NOT NULL,
    card_type VARCHAR(20),
    credit_limit NUMERIC,
    outstanding_amount NUMERIC,
    issued_date DATE,
    expiry_date DATE,
    status VARCHAR(20),
    rewards_points NUMERIC,
    branch_id INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- --------------------------
-- 8. Branches
-- --------------------------
CREATE TABLE branches (
    branch_id SERIAL PRIMARY KEY,
    branch_name VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(50),
    pincode VARCHAR(10),
    manager_name VARCHAR(100),
    contact_number VARCHAR(20),
    opened_date DATE,
    status VARCHAR(20),
    IFSC_code VARCHAR(20)
);

-- --------------------------
-- 9. Customers
-- --------------------------
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    dob DATE,
    gender VARCHAR(10),
    phone_number VARCHAR(20),
    email VARCHAR(100),
    address TEXT,
    city VARCHAR(50),
    state VARCHAR(50),
    nationality VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- --------------------------
-- 10. Transactions
-- --------------------------
CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    account_id INT NOT NULL,
    transaction_date TIMESTAMP,
    transaction_type VARCHAR(50),
    amount NUMERIC,
    balance_after NUMERIC,
    description TEXT,
    channel VARCHAR(50),
    reference_number VARCHAR(50),
    status VARCHAR(20),
    branch_id INT,
    FOREIGN KEY (account_id) REFERENCES casa_accounts(account_id)
);
