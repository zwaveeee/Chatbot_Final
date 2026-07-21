const form = document.getElementById("feedbackForm");
  const submitBtn = document.getElementById("submitBtn");
  const ratingError = document.getElementById("ratingError");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const rating = form.querySelector('input[name="rating"]:checked');
    if (!rating) {
      ratingError.classList.add("show");
      form.querySelector(".grade-slip").scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    ratingError.classList.remove("show");

    const easeEl = form.querySelector('input[name="ease_of_use"]:checked');
    const accuracyEl = form.querySelector('input[name="accuracy"]:checked');

    const payload = {
      rating: Number(rating.value),
      ease_of_use: easeEl ? easeEl.value : "",
      accuracy: accuracyEl ? accuracyEl.value : "",
      comments: form.querySelector('textarea[name="comments"]').value.trim()
    };

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Submitting...';

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok || !data.success) {
        throw new Error(data.error || "Something went wrong");
      }

      document.getElementById("formView").style.display = "none";
      document.getElementById("successView").classList.add("show");
    } catch (err) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<i class="bi bi-send-fill"></i> Submit Feedback';
      ratingError.textContent = "Couldn't submit right now -- please try again.";
      ratingError.classList.add("show");
      console.error("Feedback submit failed:", err);
    }
  });