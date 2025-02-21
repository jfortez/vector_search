import { useCallback, useRef, useState } from "react";
import { Loader2, PlusIcon, Settings2 } from "lucide-react";

import { useQuery, useQueryClient } from "@tanstack/react-query";
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
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import { toast } from "sonner";

import debounce from "lodash/debounce";
import { Button } from "./components/ui/button";
import useLicense from "./useLicense";
import listService, { type List, type Modes, type SearchParams } from "@/services/list-service";

const editSettings: EditSettingsModel = {
  allowAdding: true,
  allowDeleting: true,
  allowEditing: true,
  mode: "Normal",
  showDeleteConfirmDialog: true,
};
const pageSettings: PageSettingsModel = { pageSize: 20 };

type ActionBeginArgs = Omit<
  EditEventArgs & SaveEventArgs & AddEventArgs & DeleteEventArgs,
  "data"
> & { data: List };

const App = () => {
  useLicense();
  const [mode, setMode] = useState<Modes>("faiss");
  const [inputValue, setInputValue] = useState<string>("");
  const [threshold, setThreshold] = useState<number>(0.1);
  const { data, error, isLoading } = useQuery({
    queryKey: ["list"],
    queryFn: () => listService.getList(),
  });

  const defaultListValues = useRef<List[]>(null);
  const gridRef = useRef<GridComponent>(null);
  const client = useQueryClient();

  const handleSearch = useCallback(
    async (params: SearchParams) => {
      if (!params.search) return;
      try {
        const result = await listService.searchList({ ...params, mode });
        client.setQueryData(["list"], result);
      } catch (error) {
        const err = error as Error;
        toast.error(err.name || "Error", { description: err.message, className: "bg-destructive" });
      }
    },
    [client, mode]
  );

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const onChange = useCallback(
    debounce(async (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = event.target.value;
      if (!value) client.setQueryData(["list"], defaultListValues.current);

      setInputValue(value);
      handleSearch({ search: value, threshold });
    }, 300),
    [threshold]
  );

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const onChangeThreshold = useCallback(
    debounce(async (event: React.ChangeEvent<HTMLInputElement>) => {
      const threshold = Number(event.target.value);

      setThreshold(threshold);
      handleSearch({ threshold, search: inputValue });
    }, 300),
    [inputValue]
  );

  const actionBegin = useCallback(
    async (e: ActionBeginArgs) => {
      const { requestType, data, action } = e;
      try {
        let hasFetched = false;
        let result;

        if (requestType === "delete") {
          const [id] = (data as unknown as List[]).map((item) => item.id);
          result = await listService.deleteItem(id);
          hasFetched = true;
        }
        if (requestType === "save") {
          if (action === "edit") {
            result = await listService.updateItem(data);
            hasFetched = true;
          } else if (action === "add") {
            result = await listService.insertItem({
              nombre: data.nombre,
              identificacion: data.identificacion,
            });
            hasFetched = true;
          }
        }
        if (hasFetched) {
          client.invalidateQueries({ queryKey: ["list"] });
        }
        if (result) {
          toast.success("Success", { description: result.message });
        }
      } catch (error) {
        const err = error as Error;
        toast.error(err.name || "Error", { description: err.message });
        client.invalidateQueries({ queryKey: ["list"] });
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

  const handleModeChange = (mode: Modes) => {
    setMode(mode);
    handleSearch({ search: inputValue, mode, threshold });
  };

  return (
    <div className="min-h-dvh container mx-auto px-4 py-6 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1 w-full">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="size-9">
                <Settings2 className="size-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuLabel>Search Mode</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuCheckboxItem
                checked={mode === "faiss"}
                onCheckedChange={() => handleModeChange("faiss")}
              >
                FAISS
              </DropdownMenuCheckboxItem>

              <DropdownMenuCheckboxItem
                checked={mode === "fuzzy"}
                onCheckedChange={() => handleModeChange("fuzzy")}
              >
                Fuzzy
              </DropdownMenuCheckboxItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <div className="flex items-center gap-4 w-full">
            <Input placeholder="Filter..." onChange={onChange} className="max-w-sm" />
            <Input
              placeholder="Threshold"
              onChange={onChangeThreshold}
              className="w-[80px]"
              min={0}
              max={1}
              step={0.1}
              defaultValue={0.1}
              type="number"
            />
          </div>
        </div>

        <Button
          className="bg-green-700 text-white hover:bg-green-800 font-medium"
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
