import { useCallback, useRef } from "react";
import { Loader2, PlusIcon } from "lucide-react";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import listService, { List } from "@/services/list-service";
import {
  ColumnDirective,
  ColumnsDirective,
  GridComponent,
  Inject,
  Edit,
  Toolbar,
  CommandColumn,
  Page,
  type EditSettingsModel,
  type EditEventArgs,
  type SaveEventArgs,
  type AddEventArgs,
  type DeleteEventArgs,
  type PageSettingsModel,
} from "@syncfusion/ej2-react-grids";

import { Input } from "@/components/ui/input";

import { toast } from "sonner";

import debounce from "lodash/debounce";
import { Button } from "./components/ui/button";
import useLicense from "./useLicense";

const editSettings: EditSettingsModel = {
  allowAdding: true,
  allowDeleting: true,
  allowEditing: true,
  mode: "Normal",
  showDeleteConfirmDialog: true,
};
const pageSettings: PageSettingsModel = { pageSize: 20 };
const App = () => {
  useLicense();
  const { data, error, isLoading } = useQuery({
    queryKey: ["list"],
    queryFn: () => listService.getList(),
  });
  const defaultListValues = useRef<List[]>(null);
  const gridRef = useRef<GridComponent>(null);
  const client = useQueryClient();

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const onChange = useCallback(
    debounce(async (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = event.target.value;
      if (!value) client.setQueryData(["list"], defaultListValues.current);

      try {
        const result = await listService.searchList(value);
        client.setQueryData(["list"], result);
      } catch (error: unknown) {
        const err = error as Error;
        toast.error(err.name || "Error", { description: err.message, className: "bg-destructive" });
      }
    }, 300),
    []
  );

  const actionBegin = useCallback(
    async (e: EditEventArgs | SaveEventArgs | AddEventArgs | DeleteEventArgs) => {
      const { requestType, data, action } = e;
      console.log(e);
      try {
        let hasFetched = false;
        let result;
        if (requestType === "delete") {
          result = await listService.deleteItem(data.map((item) => item.id));

          hasFetched = true;
        }
        if (requestType === "save") {
          if (action === "edit") {
            result = await listService.updateItem(data);

            hasFetched = true;
          } else if (action === "add") {
            result = await listService.insertItem(data);

            hasFetched = true;
          }
        }
        if (hasFetched) {
          client.invalidateQueries({ queryKey: ["list"] });
          if (result) {
            toast.success("Success", { description: result.message });
          }
        }
      } catch (error) {
        const err = error as Error;
        toast.error(err.name || "Error", { description: err.message });
      }
    },
    [client]
  );

  const onAddClick = useCallback(() => {
    gridRef.current?.addRecord();
  }, []);

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  if (isLoading) {
    return (
      <div className="min-h-dvh container mx-auto px-4 flex items-center justify-center">
        <Loader2 className="animate-spin size-12 " />
      </div>
    );
  }

  if (!defaultListValues.current && data !== undefined) {
    defaultListValues.current = data;
  }

  return (
    <div className="min-h-dvh container mx-auto px-4 py-6 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <Input placeholder="Filter..." onChange={onChange} className="max-w-sm" />
        <Button
          className="bg-green-700 text-white hover:bg-green-800 font-medium pointer"
          variant="secondary"
          onClick={onAddClick}
        >
          <PlusIcon className="mr-2" />
          New Item
        </Button>
      </div>
      <GridComponent
        ref={gridRef}
        dataSource={data}
        height={500}
        editSettings={editSettings}
        pageSettings={pageSettings}
        allowPaging
        actionBegin={actionBegin}
      >
        <Inject services={[Edit, Toolbar, CommandColumn, Page]} />

        <ColumnsDirective>
          <ColumnDirective
            commands={[
              {
                type: "Edit",
                buttonOption: {
                  iconCss: "e-edit e-icons",
                  cssClass: "e-flat",
                },
              },
              {
                type: "Delete",
                buttonOption: {
                  iconCss: "e-trash e-icons",
                  cssClass: "e-flat e-danger",
                },
              },
              {
                type: "Save",
                buttonOption: {
                  iconCss: "e-save e-icons",
                  cssClass: "e-flat",
                },
              },
              {
                type: "Cancel",
                buttonOption: {
                  iconCss: "e-cancel e-icons",
                  cssClass: "e-flat",
                },
              },
            ]}
            width="150px"
          />
          <ColumnDirective
            field="id"
            headerText="ID"
            textAlign="Right"
            isPrimaryKey
            allowEditing={false}
          />
          <ColumnDirective field="identificacion" headerText="Identificación" />
          <ColumnDirective field="nombre" headerText="Nombre" />
        </ColumnsDirective>
      </GridComponent>
    </div>
  );
};

export default App;
