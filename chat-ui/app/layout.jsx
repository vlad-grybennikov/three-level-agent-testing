import "./globals.css";

export const metadata = {
  title: "telecom-aut chat",
  description:
    "Demo chat over the telecom agent under test. Plain-text replies; " +
    "step logs live in the Python server terminal and chat_logs/.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
