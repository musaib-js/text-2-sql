CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    phone_number VARCHAR(20),
    email VARCHAR(100)
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price NUMERIC
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    product_id INT REFERENCES products(id),
    order_date DATE,
    amount NUMERIC
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    position VARCHAR(50),
    store_id INT REFERENCES stores(id)
);

CREATE TABLE stores (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    city VARCHAR(100)
);

INSERT INTO customers (name, phone_number, email) VALUES
('John Doe', '1234567890', 'john@example.com'),
('Alice Smith', '9876543210', 'alice@example.com');

INSERT INTO products (name, category, price) VALUES
('Laptop', 'Electronics', 1000),
('Book', 'Education', 20);

INSERT INTO orders (customer_id, product_id, order_date, amount) VALUES
(1, 1, '2025-09-20', 1000),
(2, 2, '2025-09-21', 20);

INSERT INTO stores (name, city) VALUES
('Tech Store', 'New York'),
('Book Haven', 'Los Angeles');

INSERT INTO employees (name, position, store_id) VALUES
('Jane Doe', 'Manager', 1),
('Bob Brown', 'Sales Associate', 2);
