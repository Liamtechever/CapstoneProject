"use strict";
const chatForm = document.querySelector("[data-chat-form]");
if (chatForm) {
  chatForm.addEventListener("submit", () => {
    chatForm.querySelector("[data-send-button]").disabled = true;
    chatForm.querySelector("[data-status]").hidden = false;
  });
}
window.addEventListener("pageshow", () => {
  const button = document.querySelector("[data-send-button]");
  if (button) button.disabled = false;
  const status = document.querySelector("[data-status]");
  if (status) status.hidden = true;
});
