import React from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router";

import { queryClient } from "./client";

function Providers({ children }: { children: React.ReactNode }) {
  return (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </BrowserRouter>
  );
}

export default Providers;
