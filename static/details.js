// Get threat details from URL params
const params = new URLSearchParams(window.location.search);
const threatId = params.get("id");

// Fetch threat details from backend
async function loadThreatDetails() {
  const response = await fetch(`/api/threats/${threatId}`);
  const data = await response.json();

  // Populate UI
  document.getElementById("threat-title").textContent = data.type;
  document.getElementById(
    "threat-source"
  ).textContent = `${data.src_ip} (${data.geo_info.country})`;
  document.getElementById(
    "threat-target"
  ).textContent = `${data.dst_ip}:${data.dst_port} (${data.service})`;
  document.getElementById("threat-time").textContent = new Date(
    data.timestamp
  ).toLocaleString();
  document.getElementById(
    "threat-severity"
  ).textContent = `${data.severity} (${data.confidence}/100)`;
  document.getElementById("packet-data").textContent = formatHexDump(
    data.packet_hex
  );
}

// Action functions
function blockIP() {
  const ip = document.getElementById("threat-source").textContent.split(" ")[0];
  fetch("/api/block", {
    method: "POST",
    body: JSON.stringify({ ip: ip }),
    headers: { "Content-Type": "application/json" },
  }).then(() => alert(`${ip} blocked successfully`));
}

function ignoreThreat() {
  // Mark as false positive logic
}

function downloadPCAP() {
  // PCAP download logic
}

// Initialize
loadThreatDetails();
