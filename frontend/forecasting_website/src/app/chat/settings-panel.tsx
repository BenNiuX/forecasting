import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  GearIcon,
  MagnifyingGlassIcon,
  ChatBubbleIcon,
  ResetIcon,
  CalendarIcon,
  Cross2Icon,
} from "@radix-ui/react-icons";
import { format } from "date-fns";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { defaultImpactPrompt, defaultPlannerPrompt, defaultPublisherPrompt } from "../prompts/prompts";
import { useForecastStore } from "../store/forecastStore";

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const MODELS = {
  "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
  // "claude-3-sonnet-20240229": "Claude 3 Sonnet",
  "gpt-4o": "GPT 4o",
  // "gpt-4o-2024-08-06": "gpt-4o-2024-08-06	",
  // "gpt-4o-mini": "gpt-4o-mini",
  // "gpt-3.5-turbo-0125": "GPT 3.5 Turbo",
  // "gemini-1.5-pro": "gemini-1.5-pro", // Less quota
  // "gemini-1.5-flash": "Gemini 1.5 Flash",
  // "gemini-1.0-pro": "Gemini Pro",
  // "accounts/fireworks/models/llama-v3p1-405b-instruct": "Llama 3.1 405B",
  // "accounts/fireworks/models/llama-v3p1-70b-instruct": "Llama 3.1 70B",
  // "accounts/fireworks/models/llama-v3p1-8b-instruct": "Llama 3.1 8B"
  // Copilot
  // NIM
  // "meta/llama-3.2-3b-instruct": "Llama 3.2 3B",
  // "meta/llama-3.2-1b-instruct": "Llama 3.2 1B",
  // "meta/llama-3.1-405b-instruct": "Llama 3.1 405B",
  // "meta/llama-3.1-70b-instruct": "Llama 3.1 70B",
  // "meta/llama-3.1-8b-instruct": "Llama 3.1 8B",
  // "google/gemma-2-27b-it": "Gemma 2 27B",
  // "google/gemma-2-9b-it": "Gemma 2 9B",
  // "google/gemma-2-2b-it": "Gemma 2 2B",
  // "google/gemma-7b": "Gemma 7B",
  // "google/gemma-2b": "Gemma 2B",
  // "microsoft/phi-3.5-moe-instruct": "Phi 3.5 MOE",
  // "microsoft/phi-3.5-mini-instruct": "Phi 3.5 Mini",
  // "microsoft/phi-3-medium-128k-instruct": "Phi 3 Medium 128K",
  // "microsoft/phi-3-medium-4k-instruct": "Phi 3 Medium 4K",
  // "microsoft/phi-3-small-128k-instruct": "Phi 3 Small 128K",
  // "microsoft/phi-3-small-8k-instruct": "Phi 3 Small 8K",
  // "mediatek/breeze-7b-instruct": "Breeze 7B",
};

const SettingsPanel: React.FC<SettingsPanelProps> = ({ isOpen, onClose }) => {
  const { settings, updateSetting } = useForecastStore();

  if (!isOpen) return null;

  const handleDateSelect = (date: Date | undefined) => {
    updateSetting('beforeTimestamp', date ? Math.floor(date.getTime() / 1000) : undefined);
  };

  const resetDate = (e: React.MouseEvent) => {
    e.stopPropagation();
    updateSetting('beforeTimestamp', undefined);
  };

  return (
    <div className="fixed inset-0 bg-background p-4 overflow-y-auto z-50 md:w-[48rem] md:left-0 md:right-auto">
      <Button onClick={onClose} className="absolute right-4 top-4 md:right-2 md:top-2">
        <Cross2Icon className="h-4 w-4" />
      </Button>
      <h2 className="text-2xl font-bold mb-4 pr-8">Settings</h2>

      <div className="space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:space-x-4 space-y-4 md:space-y-0">
          <div className="flex-1 space-y-2">
            <label className="flex items-center text-sm font-medium">
              <ChatBubbleIcon className="mr-2" />
              Model
            </label>
            <Select value={settings.model} onValueChange={(value) => updateSetting('model', value)}>
              <SelectTrigger>
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(MODELS).map(([value, name]) => (
                  <SelectItem key={value} value={value}>
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex-1 space-y-2">
            <label className="flex items-center text-sm font-medium">
              <MagnifyingGlassIcon className="mr-2" />
              Breadth
            </label>
            <Input
              type="number"
              value={settings.breadth}
              // disabled={true}
              onChange={(e) => updateSetting('breadth', Number(e.target.value))}
              min={1}
              max={100}
            />
            <span className="text-xs text-gray-500">Number of search queries</span>
          </div>
        </div>

        <div className="space-y-2">
          <label className="flex items-center text-sm font-medium">
            <CalendarIcon className="mr-2" />
            Before Date
          </label>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant={"outline"}
                className={`w-full justify-start text-left font-normal ${
                  !settings.beforeTimestamp && "text-muted-foreground"
                }`}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {settings.beforeTimestamp ? (
                  <span className="flex items-center justify-between w-full">
                    {format(new Date(settings.beforeTimestamp * 1000), "PPP")}
                    <Cross2Icon
                      className="h-4 w-4 cursor-pointer"
                      onClick={resetDate}
                    />
                  </span>
                ) : (
                  <span>Pick a date</span>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0">
              <Calendar
                mode="single"
                selected={settings.beforeTimestamp ? new Date(settings.beforeTimestamp * 1000) : undefined}
                onSelect={handleDateSelect}
                initialFocus
              />
            </PopoverContent>
          </Popover>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="flex items-center text-sm font-medium">
              <GearIcon className="mr-2" />
              Planner Prompt
            </label>
            <Button
              onClick={() => updateSetting('plannerPrompt', defaultPlannerPrompt)}
              variant="outline"
              size="sm"
            >
              <ResetIcon className="mr-2" />
              Reset
            </Button>
          </div>
          <Textarea
            value={settings.plannerPrompt}
            // disabled={true}
            onChange={(e) => updateSetting('plannerPrompt', e.target.value)}
            rows={12}
            className="w-full"
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="flex items-center text-sm font-medium">
              <GearIcon className="mr-2" />
              Publisher Prompt
            </label>
            <Button
              onClick={() => updateSetting('publisherPrompt', defaultPublisherPrompt)}
              variant="outline"
              size="sm"
            >
              <ResetIcon className="mr-2" />
              Reset
            </Button>
          </div>
          <Textarea
            value={settings.publisherPrompt}
            // disabled={true}
            onChange={(e) => updateSetting('publisherPrompt', e.target.value)}
            rows={12}
            className="w-full"
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="flex items-center text-sm font-medium">
              <GearIcon className="mr-2" />
              Impact Prompt
            </label>
            <Button
              onClick={() => updateSetting('impactPrompt', defaultImpactPrompt)}
              variant="outline"
              size="sm"
            >
              <ResetIcon className="mr-2" />
              Reset
            </Button>
          </div>
          <Textarea
            value={settings.impactPrompt}
            // disabled={true}
            onChange={(e) => updateSetting('impactPrompt', e.target.value)}
            rows={12}
            className="w-full"
          />
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel;