const API_URL = "http://127.0.0.1:8000";

let currentCustomerId = null;


// API request function
async function apiRequest(endpoint, options = {}) {

    const response = await fetch(
        API_URL + endpoint,
        {
            headers: {
                "Content-Type": "application/json"
            },
            ...options
        }
    );

    const data = await response.json();

    if (!response.ok) {
        throw new Error(
            data.detail || "Something went wrong"
        );
    }

    return data;
}


// Get form values
function getCustomerData() {

    const form = document.getElementById("customerForm");

    const data = new FormData(form);

    return {

        age: Number(data.get("age")),
        job: data.get("job"),
        marital: data.get("marital"),
        education: data.get("education"),
        default: data.get("default"),
        balance: Number(data.get("balance")),
        housing: data.get("housing"),
        loan: data.get("loan"),
        contact: data.get("contact"),
        day: Number(data.get("day")),
        month: data.get("month"),
        campaign: Number(data.get("campaign")),
        pdays: Number(data.get("pdays")),
        previous: Number(data.get("previous")),
        poutcome: data.get("poutcome")

    };
}


// Save customer
document.getElementById("customerForm")
    .addEventListener("submit", async function(event) {

        event.preventDefault();

        const message = document.getElementById(
            "customerMessage"
        );

        try {

            const customer = await apiRequest(
                "/customers",
                {
                    method: "POST",
                    body: JSON.stringify(getCustomerData())
                }
            );

            currentCustomerId = customer.id;

            message.textContent =
                "Customer saved. Customer ID: " + customer.id;

            await loadCustomers();

        } catch (error) {

            message.textContent = error.message;

        }
    });


// Generate prediction
document.getElementById("predictButton")
    .addEventListener("click", async function() {

        const result = document.getElementById(
            "predictionResult"
        );

        result.textContent = "Generating prediction...";

        try {

            const payload = getCustomerData();

            if (currentCustomerId !== null) {
                payload.customer_id = currentCustomerId;
            }

            const output = await apiRequest(
                "/predict",
                {
                    method: "POST",
                    body: JSON.stringify(payload)
                }
            );

            result.innerHTML = `
                <strong>Prediction:</strong>
                ${output.prediction}
                <br>
                <strong>Probability:</strong>
                ${(output.probability * 100).toFixed(2)}%
                <br>
                <strong>Model:</strong>
                ${output.model_name}
            `;

            await loadPredictions();

        } catch (error) {

            result.textContent = error.message;

        }
    });


// Display customers
async function loadCustomers() {

    const data = await apiRequest("/customers?limit=20");

    const table = document.getElementById(
        "customersTable"
    );

    table.innerHTML = "";

    data.forEach(customer => {

        const row = document.createElement("tr");

        const values = [
            customer.id,
            customer.age,
            customer.job,
            customer.balance,
            customer.housing,
            customer.loan
        ];

        values.forEach(value => {

            const cell = document.createElement("td");

            cell.textContent = value ?? "";

            row.appendChild(cell);

        });

        table.appendChild(row);

    });
}


// Display predictions
async function loadPredictions() {

    const predictions = await apiRequest("/predictions");

    const table = document.getElementById(
        "predictionsTable"
    );

    table.innerHTML = "";

    predictions.forEach(item => {

        const row = document.createElement("tr");

        const values = [
            item.id,
            item.customer_id ?? "Not linked",
            item.prediction,
            (item.probability * 100).toFixed(2) + "%",
            item.model_name,
            item.created_at
        ];

        values.forEach(value => {

            const cell = document.createElement("td");

            cell.textContent = value ?? "";

            row.appendChild(cell);

        });

        table.appendChild(row);

    });
}


// Refresh buttons
document.getElementById("refreshCustomers")
    .addEventListener("click", () => {
        loadCustomers().catch(console.error);
    });

document.getElementById("refreshPredictions")
    .addEventListener("click", () => {
        loadPredictions().catch(console.error);
    });


// Check API connection
async function checkApi() {

    const status = document.getElementById("apiStatus");

    try {

        await apiRequest("/customers?limit=1");

        status.textContent = "API Connected";

    } catch (error) {

        status.textContent = "API Not Connected";

    }
}


// Load dashboard data
async function startDashboard() {

    await checkApi();

    try {

        await loadCustomers();

        await loadPredictions();

    } catch (error) {

        console.error(error);

    }
}


startDashboard();