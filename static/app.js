// Wait for full DOM load
document.addEventListener("DOMContentLoaded", function () {
  // 1. Get canvas element
  const canvas = document.getElementById("threatChart");

  // 2. Verify element exists
  if (!canvas) {
    console.error("Canvas element not found! Check HTML ID");
    return;
  }

  // 3. Initialize chart
  const ctx = canvas.getContext("2d");
  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Threats Detected",
          data: [],
          borderColor: "#64ffda",
          backgroundColor: "rgba(100, 255, 218, 0.1)",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
    },
  });

  // 4. WebSocket connection
  const socket = io();

  socket.on("threat_update", (data) => {
    chart.data.labels = data.labels;
    chart.data.datasets[0].data = data.data;
    chart.update();
  });

  // 5. Initial data load
  fetch("/threat_data")
    .then((response) => response.json())
    .then((data) => {
      chart.data.labels = data.labels;
      chart.data.datasets[0].data = data.data;
      chart.update();
    })
    .catch((error) => console.error("Data load failed:", error));
});
