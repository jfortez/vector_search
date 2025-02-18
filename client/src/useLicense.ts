import { useEffect, useRef } from "react";

const useLicense = () => {
  const observer = useRef(null);

  useEffect(() => {
    const cto = document.querySelector('a[href*="syncfusion"]')?.closest("div");
    if (cto) {
      cto.style.display = "none";
    }
  }, []);

  useEffect(() => {
    const handleDocumentChange = () => {
      const bodyNodes = document.body.childNodes;
      bodyNodes.forEach((el) => {
        if (el.nodeName === "DIV" && !el.id && el.querySelector('a[href*="syncfusion"]')) {
          el.style.display = "none";
        }
      });
    };

    const timer = setTimeout(() => {
      observer.current = new MutationObserver(handleDocumentChange);

      observer.current.observe(document.body, { childList: true });
    });

    return () => {
      clearTimeout(timer);
    };
  }, []);
  return null;
};

export default useLicense;
