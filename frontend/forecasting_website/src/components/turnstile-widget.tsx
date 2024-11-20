import React from "react";
import { Turnstile } from "@marsidev/react-turnstile";

interface TurnstileWidgetProps {
  setToken: React.Dispatch<React.SetStateAction<string | null>>;
}

export default function TurnstileWidget({ setToken }: TurnstileWidgetProps) {
  console.log(
    "2NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY:",
    process.env.NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY
  );
  console.log(
    "APPSETTING_NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY:",
    process.env.APPSETTING_NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY
  );
  let siteKey = process.env.NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY || "";
  if (siteKey === "") {
    siteKey = "1x00000000000000000000AA";
  }

  return (
    <Turnstile
      siteKey={siteKey}
      onSuccess={setToken}
      options={{
        // size: 'invisible',
        theme: "auto"
      }}
    />
  );
}
